// ─────────────────────────────────────────────────────────────────────────────
// RAG Benchmark Orchestrator
//
// This Go program does one job: send queries to the Python RAG servers in
// parallel, collect results, and expose real-time metrics to Prometheus.
//
// WHY GO INSTEAD OF PYTHON?
// ─────────────────────────────────────────────────────────────────────────────
// Python has the GIL (Global Interpreter Lock) — only one thread runs at a
// time. To send 100 concurrent requests, you need asyncio (complex) or
// multiprocessing (heavy). Go has goroutines — lightweight threads that cost
// ~2KB each. You can launch 10,000 of them trivially.
//
// ARCHITECTURE:
// ─────────────────────────────────────────────────────────────────────────────
//
//   Python servers (one per framework, each with index pre-built):
//     localhost:8100  ←  LangChain  (Chroma + bge-m3 + Llama-3.1-8B)
//     localhost:8101  ←  LlamaIndex (in-memory + bge-m3 + Llama-3.1-8B)
//     localhost:8102  ←  DSPy       (FAISS + bge-m3 + Llama-3.1-8B)
//
//   Go orchestrator:
//     reads queries.json  →  routes each query to the right server
//     goroutine 1  →  POST localhost:8100/query  { question: "..." }
//     goroutine 2  →  POST localhost:8100/query  { question: "..." }  ← same time
//     goroutine 3  →  POST localhost:8101/query  { question: "..." }  ← same time
//     goroutine N  →  POST localhost:8102/query  { question: "..." }  ← same time
//
//   The GPU stays saturated because multiple requests are in-flight at once.
//   vLLM handles them via continuous batching — it groups concurrent requests
//   into a single forward pass instead of processing them one by one.
//
// ─────────────────────────────────────────────────────────────────────────────
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"golang.org/x/time/rate"
)

// httpClient is shared across all goroutines. The 5-minute timeout covers the
// slowest possible LLM generation — if a request exceeds this the server is
// hung and the goroutine should be freed rather than waiting forever.
var httpClient = &http.Client{Timeout: 5 * time.Minute}

// ─────────────────────────────────────────────────────────────────────────────
// Data types
// ─────────────────────────────────────────────────────────────────────────────

// Query is one row from the input queries.json file.
// The framework field tells us which Python server to route to.
type Query struct {
	Question    string `json:"question"`
	GroundTruth string `json:"ground_truth"`
	Domain      string `json:"domain"`
	Framework   string `json:"framework"` // "langchain", "llamaindex", or "dspy"
}

// RAGRequest is the JSON body we POST to /query on each Python server.
// Matches QueryRequest in src/rag_server.py.
type RAGRequest struct {
	Question    string `json:"question"`
	Domain      string `json:"domain"`
	GroundTruth string `json:"ground_truth"`
}

// RAGResponse is what the Python server returns from POST /query.
// Matches QueryResponse in src/rag_server.py.
type RAGResponse struct {
	Question      string   `json:"question"`
	GroundTruth   string   `json:"ground_truth"`
	Domain        string   `json:"domain"`
	Framework     string   `json:"framework"`
	Answer        string   `json:"answer"`
	Contexts      []string `json:"contexts"`
	RetrievedNoise bool    `json:"retrieved_noise"`
	RetrievalMs   float64  `json:"retrieval_ms"`
	GenerationMs  float64  `json:"generation_ms"`
	LatencyMs     float64  `json:"latency_ms"`
	Error         string   `json:"error"`
}

// Result is what we write to the output JSON file.
type Result struct {
	Question      string   `json:"question"`
	GroundTruth   string   `json:"ground_truth"`
	Domain        string   `json:"domain"`
	Framework     string   `json:"framework"`
	Answer        string   `json:"answer"`
	Contexts      []string `json:"contexts"`
	RetrievedNoise bool    `json:"retrieved_noise"`
	RetrievalMs   float64  `json:"retrieval_ms"`
	GenerationMs  float64  `json:"generation_ms"`
	LatencyMs     float64  `json:"latency_ms"`
	Error         string   `json:"error,omitempty"`
}

// ─────────────────────────────────────────────────────────────────────────────
// Prometheus metrics
// ─────────────────────────────────────────────────────────────────────────────

var (
	queryDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "rag_query_duration_seconds",
			Help:    "End-to-end query latency in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"framework"},
	)

	// Now populated with real retrieval time from the Python server response.
	retrievalDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "rag_retrieval_duration_seconds",
			Help:    "Retrieval step latency in seconds (embed + vector search)",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"framework"},
	)

	generationDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "rag_generation_duration_seconds",
			Help:    "LLM generation step latency in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"framework"},
	)

	queriesTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "rag_queries_total",
			Help: "Total number of queries processed",
		},
		[]string{"framework", "status"},
	)

	errorsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "rag_errors_total",
			Help: "Total number of query errors",
		},
		[]string{"framework"},
	)

	activeWorkers = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "rag_active_workers",
			Help: "Number of worker goroutines currently processing a query",
		},
	)

	noisePoisonTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "rag_noise_poison_total",
			Help: "Number of queries where a noise document was retrieved",
		},
		[]string{"framework"},
	)
)

func init() {
	prometheus.MustRegister(queryDuration)
	prometheus.MustRegister(retrievalDuration)
	prometheus.MustRegister(generationDuration)
	prometheus.MustRegister(queriesTotal)
	prometheus.MustRegister(errorsTotal)
	prometheus.MustRegister(activeWorkers)
	prometheus.MustRegister(noisePoisonTotal)
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

func main() {
	inputFile   := flag.String("input",       "queries.json",  "Path to input JSON file with queries")
	outputFile  := flag.String("output",      "results.json",  "Path to write results JSON")
	workers     := flag.Int("workers",        8,               "Number of concurrent worker goroutines")
	rps         := flag.Float64("rps",        10,              "Max requests per second (rate limiter)")
	burst       := flag.Int("burst",          20,              "Rate limiter burst size")
	metricsPort := flag.String("metrics-port","9091",          "Port for Prometheus metrics server")

	// One URL flag per framework — defaults to localhost ports set in run_servers.sh
	langchainURL  := flag.String("langchain-url",  "http://localhost:8100", "LangChain server URL")
	llamaindexURL := flag.String("llamaindex-url", "http://localhost:8101", "LlamaIndex server URL")
	dspyURL       := flag.String("dspy-url",       "http://localhost:8102", "DSPy server URL")

	flag.Parse()

	// Map framework name → server URL. executeQuery uses this to route each query.
	serverURLs := map[string]string{
		"langchain":  *langchainURL,
		"llamaindex": *llamaindexURL,
		"dspy":       *dspyURL,
	}

	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("  RAG Benchmark Orchestrator (Go)")
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Printf("  Input:        %s\n", *inputFile)
	fmt.Printf("  Output:       %s\n", *outputFile)
	fmt.Printf("  Workers:      %d goroutines\n", *workers)
	fmt.Printf("  Rate limit:   %.1f req/sec (burst: %d)\n", *rps, *burst)
	fmt.Printf("  LangChain:    %s\n", *langchainURL)
	fmt.Printf("  LlamaIndex:   %s\n", *llamaindexURL)
	fmt.Printf("  DSPy:         %s\n", *dspyURL)
	fmt.Printf("  Metrics:      http://localhost:%s/metrics\n", *metricsPort)
	fmt.Println("═══════════════════════════════════════════════════════════")

	// ── Graceful shutdown ────────────────────────────────────────────────────
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigChan
		fmt.Printf("\n  Received %s — shutting down gracefully...\n", sig)
		cancel()
	}()

	// ── Prometheus metrics server ────────────────────────────────────────────
	go func() {
		mux := http.NewServeMux()
		mux.Handle("/metrics", promhttp.Handler())
		srv := &http.Server{Addr: ":" + *metricsPort, Handler: mux}
		fmt.Printf("  Metrics server listening on :%s\n", *metricsPort)
		if err := srv.ListenAndServe(); err != http.ErrServerClosed {
			log.Printf("  Metrics server error: %v", err)
		}
	}()

	// ── Health check all servers before sending queries ──────────────────────
	// If a server isn't ready (index still building), we wait here rather than
	// flooding it with queries that will get 503s.
	fmt.Println("\n  Waiting for RAG servers to be ready...")
	for name, url := range serverURLs {
		waitForReady(ctx, name, url)
	}
	fmt.Println("  All servers ready.\n")

	// ── Load queries ─────────────────────────────────────────────────────────
	queries, err := loadQueries(*inputFile)
	if err != nil {
		log.Fatalf("  Failed to load queries: %v", err)
	}
	fmt.Printf("  Loaded %d queries\n", len(queries))

	// ── Rate limiter ─────────────────────────────────────────────────────────
	limiter := rate.NewLimiter(rate.Limit(*rps), *burst)

	// ── Channels ─────────────────────────────────────────────────────────────
	queryChan  := make(chan Query)
	resultChan := make(chan Result, len(queries))

	// ── Launch workers ───────────────────────────────────────────────────────
	var wg sync.WaitGroup
	fmt.Printf("  Launching %d workers...\n", *workers)
	for i := 0; i < *workers; i++ {
		wg.Add(1)
		go worker(ctx, queryChan, resultChan, limiter, serverURLs, &wg)
	}

	// ── Feed queries ─────────────────────────────────────────────────────────
	go func() {
		for _, q := range queries {
			select {
			case queryChan <- q:
			case <-ctx.Done():
				close(queryChan)
				return
			}
		}
		close(queryChan)
	}()

	// ── Collect results ──────────────────────────────────────────────────────
	wg.Wait()
	close(resultChan)

	var results []Result
	for r := range resultChan {
		results = append(results, r)
	}

	if err := writeResults(*outputFile, results); err != nil {
		log.Fatalf("  Failed to write results: %v", err)
	}

	printSummary(results)
}

// ─────────────────────────────────────────────────────────────────────────────
// Worker
// ─────────────────────────────────────────────────────────────────────────────

func worker(
	ctx        context.Context,
	queries    <-chan Query,
	results    chan<- Result,
	limiter    *rate.Limiter,
	serverURLs map[string]string,
	wg         *sync.WaitGroup,
) {
	defer wg.Done()

	for q := range queries {
		select {
		case <-ctx.Done():
			return
		default:
		}

		if err := limiter.Wait(ctx); err != nil {
			return
		}

		activeWorkers.Inc()
		result := executeQuery(ctx, q, serverURLs)
		activeWorkers.Dec()

		fw := result.Framework
		if fw == "" {
			fw = "unknown"
		}

		// Record split latency — retrieval and generation are now real values
		// from the Python server, not approximations.
		queryDuration.WithLabelValues(fw).Observe(result.LatencyMs / 1000.0)
		retrievalDuration.WithLabelValues(fw).Observe(result.RetrievalMs / 1000.0)
		generationDuration.WithLabelValues(fw).Observe(result.GenerationMs / 1000.0)

		if result.Error != "" {
			queriesTotal.WithLabelValues(fw, "error").Inc()
			errorsTotal.WithLabelValues(fw).Inc()
		} else {
			queriesTotal.WithLabelValues(fw, "success").Inc()
		}

		if result.RetrievedNoise {
			noisePoisonTotal.WithLabelValues(fw).Inc()
		}

		results <- result
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// executeQuery — POSTs to the right Python RAG server and returns the result
// ─────────────────────────────────────────────────────────────────────────────

func executeQuery(ctx context.Context, q Query, serverURLs map[string]string) Result {
	result := Result{
		Question:    q.Question,
		GroundTruth: q.GroundTruth,
		Domain:      q.Domain,
		Framework:   q.Framework,
	}

	// Route to the right server based on the framework field in the query.
	serverURL, ok := serverURLs[q.Framework]
	if !ok {
		result.Error = fmt.Sprintf("unknown framework %q — add it to --langchain-url/--llamaindex-url/--dspy-url", q.Framework)
		return result
	}

	// Build the request body that matches RAGRequest in src/rag_server.py.
	ragReq := RAGRequest{
		Question:    q.Question,
		Domain:      q.Domain,
		GroundTruth: q.GroundTruth,
	}
	body, err := json.Marshal(ragReq)
	if err != nil {
		result.Error = fmt.Sprintf("marshal error: %v", err)
		return result
	}

	url := serverURL + "/query"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		result.Error = fmt.Sprintf("request creation error: %v", err)
		return result
	}
	req.Header.Set("Content-Type", "application/json")

	start := time.Now()
	resp, err := httpClient.Do(req)
	if err != nil {
		result.Error = fmt.Sprintf("http error: %v", err)
		result.LatencyMs = float64(time.Since(start).Milliseconds())
		return result
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		result.Error = fmt.Sprintf("read body error: %v", err)
		return result
	}

	if resp.StatusCode != http.StatusOK {
		result.Error = fmt.Sprintf("server returned status %d: %s", resp.StatusCode, string(respBody))
		return result
	}

	var ragResp RAGResponse
	if err := json.Unmarshal(respBody, &ragResp); err != nil {
		result.Error = fmt.Sprintf("unmarshal error: %v", err)
		return result
	}

	result.Answer        = ragResp.Answer
	result.Contexts      = ragResp.Contexts
	result.RetrievedNoise = ragResp.RetrievedNoise
	result.RetrievalMs   = ragResp.RetrievalMs
	result.GenerationMs  = ragResp.GenerationMs
	result.LatencyMs     = ragResp.LatencyMs
	result.Error         = ragResp.Error

	// Use the framework name the Python server reports (e.g. "dspy_optimized")
	// rather than the routing key from the query, so the Prometheus labels are accurate.
	if ragResp.Framework != "" {
		result.Framework = ragResp.Framework
	}

	return result
}

// ─────────────────────────────────────────────────────────────────────────────
// waitForReady — polls GET /health until the server says it's ready
// ─────────────────────────────────────────────────────────────────────────────

func waitForReady(ctx context.Context, name, baseURL string) {
	url := baseURL + "/health"
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err == nil {
			resp, err := httpClient.Do(req)
			if err == nil {
				var body map[string]interface{}
				if json.NewDecoder(resp.Body).Decode(&body) == nil {
					resp.Body.Close()
					if ready, ok := body["ready"].(bool); ok && ready {
						fmt.Printf("  [%s] ready ✓\n", name)
						return
					}
				} else {
					resp.Body.Close()
				}
			}
		}

		fmt.Printf("  [%s] not ready yet — retrying in 3s...\n", name)
		select {
		case <-ctx.Done():
			return
		case <-time.After(3 * time.Second):
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

func loadQueries(path string) ([]Query, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read file %s: %w", path, err)
	}
	var queries []Query
	if err := json.Unmarshal(data, &queries); err != nil {
		return nil, fmt.Errorf("parse JSON: %w", err)
	}
	return queries, nil
}

func writeResults(path string, results []Result) error {
	data, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal results: %w", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("write file %s: %w", path, err)
	}
	fmt.Printf("  Results written to %s (%d entries)\n", path, len(results))
	return nil
}

func printSummary(results []Result) {
	fmt.Println("\n═══════════════════════════════════════════════════════════")
	fmt.Println("  Summary")
	fmt.Println("═══════════════════════════════════════════════════════════")

	type stats struct {
		count, errors      int
		totalMs, retrievalMs, generationMs float64
		noiseHits          int
	}
	byFramework := make(map[string]*stats)

	for _, r := range results {
		fw := r.Framework
		if fw == "" {
			fw = "unknown"
		}
		if byFramework[fw] == nil {
			byFramework[fw] = &stats{}
		}
		s := byFramework[fw]
		s.count++
		s.totalMs += r.LatencyMs
		s.retrievalMs += r.RetrievalMs
		s.generationMs += r.GenerationMs
		if r.Error != "" {
			s.errors++
		}
		if r.RetrievedNoise {
			s.noiseHits++
		}
	}

	for fw, s := range byFramework {
		avg := func(total float64) float64 {
			if s.count == 0 {
				return 0
			}
			return total / float64(s.count)
		}
		fmt.Printf("  %-20s  queries: %4d  errors: %3d  noise_hits: %3d  retrieval: %7.1fms  generation: %7.1fms  total: %7.1fms\n",
			fw, s.count, s.errors, s.noiseHits,
			avg(s.retrievalMs), avg(s.generationMs), avg(s.totalMs))
	}

	fmt.Printf("\n  Total: %d queries processed\n", len(results))
	fmt.Println("═══════════════════════════════════════════════════════════")
}
