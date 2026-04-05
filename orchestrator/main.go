// ─────────────────────────────────────────────────────────────────────────────
// RAG Benchmark Orchestrator
//
// This Go program does one job: send queries to a vLLM server in parallel,
// collect results, and expose real-time metrics to Prometheus.
//
// WHY GO INSTEAD OF PYTHON?
// ─────────────────────────────────────────────────────────────────────────────
// Python has the GIL (Global Interpreter Lock) — only one thread runs at a
// time. To send 100 concurrent requests, you need asyncio (complex) or
// multiprocessing (heavy). Go has goroutines — lightweight threads that cost
// ~2KB each. You can launch 10,000 of them trivially.
//
// WHAT THIS PROGRAM DOES:
// ─────────────────────────────────────────────────────────────────────────────
// 1. Reads a JSON file of questions (from the Python benchmark)
// 2. Spins up a pool of N worker goroutines
// 3. Each worker picks a question, sends it to the vLLM server, times the response
// 4. A token-bucket rate limiter prevents exceeding the API rate limit
// 5. Results are written to a JSON output file
// 6. Prometheus metrics are exposed on port 9091 the entire time
//
// ARCHITECTURE:
// ─────────────────────────────────────────────────────────────────────────────
//
//   main() ──→ reads questions from JSON file
//      │
//      ├──→ starts Prometheus metrics server on :9091
//      │
//      ├──→ creates a rate limiter (e.g., 10 requests/second)
//      │
//      ├──→ launches N worker goroutines
//      │       each worker:
//      │         1. waits for rate limiter token
//      │         2. picks next question from channel
//      │         3. calls /v1/chat/completions on the vLLM server
//      │         4. measures retrieval + generation time
//      │         5. sends result back through results channel
//      │
//      └──→ collects all results and writes to output JSON
//
// ─────────────────────────────────────────────────────────────────────────────
package main

// ─────────────────────────────────────────────────────────────────────────────
// Imports — each one explained
// ─────────────────────────────────────────────────────────────────────────────

import (
	// "bytes" lets us build request bodies (JSON payloads) in memory.
	// We use bytes.NewReader() to wrap JSON bytes into something
	// http.NewRequest() can read.
	"bytes"

	// "context" provides cancellation and timeout support.
	// When you hit Ctrl+C, the context gets cancelled and all goroutines
	// can cleanly shut down instead of leaving zombie connections.
	"context"

	// "encoding/json" converts Go structs ↔ JSON strings.
	// We use it to read the input questions file and write the output results.
	"encoding/json"

	// "flag" parses command-line arguments like --workers 8 --rps 10.
	// Similar to Python's argparse but simpler — one function per flag.
	"flag"

	// "fmt" is Go's printf — formatted string output.
	"fmt"

	// "io" provides basic I/O primitives. We use io.ReadAll() to read
	// the entire HTTP response body into a byte slice.
	"io"

	// "log" adds timestamps to printed messages. log.Fatal() prints and exits.
	"log"

	// "net/http" is Go's built-in HTTP client AND server.
	// We use it both to call the vLLM API (client) and to serve
	// Prometheus metrics (server).
	"net/http"

	// "os" provides file operations and access to os.Args, os.Exit, etc.
	"os"

	// "os/signal" lets us catch Ctrl+C (SIGINT) and SIGTERM so we can
	// shut down gracefully — flush metrics, close connections, write results.
	"os/signal"

	// "sync" provides synchronization primitives. We use sync.WaitGroup
	// to wait for all worker goroutines to finish before exiting.
	"sync"

	// "syscall" gives access to OS-level signal constants like SIGTERM.
	"syscall"

	// "time" provides time measurement (time.Now(), time.Since()) and
	// durations. We use it to measure query latency.
	"time"

	// ── Third-party imports ──────────────────────────────────────────

	// prometheus client library — the standard way to expose metrics
	// that Prometheus can scrape. We define counters, histograms, and
	// gauges, and the library handles the /metrics HTTP endpoint.
	"github.com/prometheus/client_golang/prometheus"

	// promhttp provides the HTTP handler for /metrics endpoint.
	// Prometheus scrapes this endpoint every 15 seconds (configured
	// in infra/prometheus.yml) to collect our metrics.
	"github.com/prometheus/client_golang/prometheus/promhttp"

	// rate provides a token-bucket rate limiter.
	// Token bucket = imagine a bucket that gets N tokens per second.
	// Each request takes one token. If the bucket is empty, you wait.
	// This prevents us from overwhelming the vLLM server.
	"golang.org/x/time/rate"
)

// ─────────────────────────────────────────────────────────────────────────────
// Data types — Go structs (like Python dataclasses)
// ─────────────────────────────────────────────────────────────────────────────

// Query represents one question from the input JSON file.
// The `json:"..."` tags tell encoding/json which JSON key maps to which field.
// For example, `json:"question"` means the JSON key "question" maps to the
// Question field in Go.
type Query struct {
	Question   string `json:"question"`    // the question text
	GroundTruth string `json:"ground_truth"` // the expected answer
	Domain     string `json:"domain"`      // e.g., "techqa", "finqa", "covidqa"
	Framework  string `json:"framework"`   // which framework to query: "langchain", "llamaindex", "dspy"
	Context    string `json:"context"`     // retrieved context (if pre-filled)
}

// Result represents the output for one query — the answer and timing info.
type Result struct {
	Question      string  `json:"question"`
	GroundTruth   string  `json:"ground_truth"`
	Domain        string  `json:"domain"`
	Framework     string  `json:"framework"`
	Answer        string  `json:"answer"`
	TotalMs       float64 `json:"total_ms"`       // end-to-end time
	GenerationMs  float64 `json:"generation_ms"`  // just the LLM call time
	Error         string  `json:"error,omitempty"` // omitempty = don't include if empty
}

// ChatRequest is the JSON body we send to the vLLM /v1/chat/completions API.
// This matches the OpenAI chat completions format — vLLM is compatible.
type ChatRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`
}

// ChatMessage is one message in the conversation (system, user, or assistant).
type ChatMessage struct {
	Role    string `json:"role"`    // "system", "user", or "assistant"
	Content string `json:"content"` // the message text
}

// ChatResponse is what vLLM returns. We only care about the first choice's
// message content — the actual answer text.
type ChatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

// ─────────────────────────────────────────────────────────────────────────────
// Prometheus metrics — defined at package level so all goroutines can access them
// ─────────────────────────────────────────────────────────────────────────────
//
// Prometheus has three main metric types:
//   - Counter: a number that only goes UP (total queries, total errors)
//   - Histogram: records a distribution of values (latency in seconds)
//   - Gauge: a number that goes UP and DOWN (active workers right now)
//
// Each metric has labels (like tags) that let you filter. For example,
// rag_queries_total{framework="langchain"} counts only LangChain queries.
// These metric names match the Grafana dashboard panels in
// infra/grafana/dashboards/rag_benchmark.json — once this program runs,
// those "No data" panels will light up.

var (
	// queryDuration records how long each query takes end-to-end.
	// It's a Histogram because we want percentiles (p50, p95, p99).
	// Buckets define the boundaries: 0.05s, 0.1s, 0.25s, etc.
	// Prometheus counts how many observations fall into each bucket,
	// then Grafana uses histogram_quantile() to calculate percentiles.
	queryDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "rag_query_duration_seconds",
			Help:    "End-to-end query latency in seconds",
			Buckets: prometheus.DefBuckets, // [.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10]
		},
		[]string{"framework"}, // label: which framework this query was for
	)

	// retrievalDuration records just the retrieval step (finding documents).
	// In the Go orchestrator, we don't actually do retrieval — the Python
	// pipelines do that. But when the Python benchmark calls the Go
	// orchestrator's API, it can report retrieval time separately.
	// For now, this measures the full call since vLLM doesn't split
	// retrieval from generation — we record it as generation only.
	retrievalDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "rag_retrieval_duration_seconds",
			Help:    "Retrieval step latency in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"framework"},
	)

	// generationDuration records just the LLM generation step.
	generationDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "rag_generation_duration_seconds",
			Help:    "LLM generation step latency in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"framework"},
	)

	// queriesTotal counts the total number of queries processed.
	// It's a Counter — it only goes up, never down.
	// Labels: framework + status ("success" or "error").
	queriesTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "rag_queries_total",
			Help: "Total number of queries processed",
		},
		[]string{"framework", "status"},
	)

	// errorsTotal counts the total number of failed queries.
	// Separate from queriesTotal for convenience — Grafana can show
	// error rate as rate(rag_errors_total) / rate(rag_queries_total).
	errorsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "rag_errors_total",
			Help: "Total number of query errors",
		},
		[]string{"framework"},
	)

	// activeWorkers tracks how many goroutines are currently processing
	// a query. It's a Gauge — goes up when a worker starts a query,
	// goes down when it finishes. Useful for seeing if all workers are
	// busy (saturated) or mostly idle.
	activeWorkers = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "rag_active_workers",
			Help: "Number of worker goroutines currently processing a query",
		},
	)
)

// init() runs automatically before main(). We use it to register all
// our metrics with Prometheus. If you forget this step, the metrics
// won't appear on the /metrics endpoint.
func init() {
	prometheus.MustRegister(queryDuration)
	prometheus.MustRegister(retrievalDuration)
	prometheus.MustRegister(generationDuration)
	prometheus.MustRegister(queriesTotal)
	prometheus.MustRegister(errorsTotal)
	prometheus.MustRegister(activeWorkers)
}

// ─────────────────────────────────────────────────────────────────────────────
// Main function — entry point
// ─────────────────────────────────────────────────────────────────────────────

func main() {
	// ── Parse command-line flags ─────────────────────────────────────
	//
	// flag.String/Int/Float64 define flags with:
	//   1. flag name (what the user types: --input)
	//   2. default value
	//   3. help text (shown with --help)
	//
	// flag.Parse() reads os.Args and fills in the values.
	// The * before each variable dereferences the pointer — flag functions
	// return *string, *int, etc. (pointers), so we dereference once.

	inputFile := flag.String("input", "queries.json", "Path to input JSON file with queries")
	outputFile := flag.String("output", "results.json", "Path to write results JSON")
	workers := flag.Int("workers", 8, "Number of concurrent worker goroutines")
	rps := flag.Float64("rps", 10, "Max requests per second (rate limiter)")
	burst := flag.Int("burst", 20, "Rate limiter burst size (max concurrent above steady rate)")
	vllmURL := flag.String("vllm-url", "http://localhost:8000", "vLLM server base URL")
	model := flag.String("model", "meta-llama/Meta-Llama-3-8B-Instruct", "Model name for vLLM")
	metricsPort := flag.String("metrics-port", "9091", "Port for Prometheus metrics server")

	flag.Parse()

	// Print the configuration so the user can verify what's running
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("  RAG Benchmark Orchestrator (Go)")
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Printf("  Input:        %s\n", *inputFile)
	fmt.Printf("  Output:       %s\n", *outputFile)
	fmt.Printf("  Workers:      %d goroutines\n", *workers)
	fmt.Printf("  Rate limit:   %.1f req/sec (burst: %d)\n", *rps, *burst)
	fmt.Printf("  vLLM server:  %s\n", *vllmURL)
	fmt.Printf("  Model:        %s\n", *model)
	fmt.Printf("  Metrics:      http://localhost:%s/metrics\n", *metricsPort)
	fmt.Println("═══════════════════════════════════════════════════════════")

	// ── Set up graceful shutdown ─────────────────────────────────────
	//
	// context.WithCancel() creates a context that can be cancelled.
	// When cancelled, all goroutines watching this context will stop.
	//
	// signal.Notify tells Go: "when the OS sends SIGINT (Ctrl+C) or
	// SIGTERM (docker stop), put a message on the 'sigChan' channel."
	//
	// We then watch sigChan in a goroutine — when a signal arrives,
	// we call cancel() which cancels the context, which tells all
	// workers to stop.

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel() // ensure cancel is called when main() returns

	sigChan := make(chan os.Signal, 1)                      // buffered channel, holds 1 signal
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM) // subscribe to signals

	// Watch for shutdown signal in a background goroutine
	go func() {
		sig := <-sigChan // blocks until a signal arrives
		fmt.Printf("\n  Received %s — shutting down gracefully...\n", sig)
		cancel() // cancel the context → all workers will notice and stop
	}()

	// ── Start Prometheus metrics server ──────────────────────────────
	//
	// This runs an HTTP server in a background goroutine.
	// Prometheus scrapes http://localhost:9091/metrics every 15 seconds
	// (configured in infra/prometheus.yml under job "go-orchestrator").
	//
	// promhttp.Handler() returns an HTTP handler that formats all
	// registered metrics in Prometheus text format — lines like:
	//   rag_queries_total{framework="langchain",status="success"} 42
	//   rag_active_workers 3

	go func() {
		mux := http.NewServeMux()              // create a new HTTP router
		mux.Handle("/metrics", promhttp.Handler()) // register /metrics endpoint
		metricsServer := &http.Server{
			Addr:    ":" + *metricsPort,
			Handler: mux,
		}
		fmt.Printf("  Metrics server listening on :%s\n", *metricsPort)
		if err := metricsServer.ListenAndServe(); err != http.ErrServerClosed {
			log.Printf("  Metrics server error: %v", err)
		}
	}()

	// ── Load queries from JSON file ─────────────────────────────────

	queries, err := loadQueries(*inputFile)
	if err != nil {
		log.Fatalf("  Failed to load queries: %v", err)
	}
	fmt.Printf("  Loaded %d queries\n", len(queries))

	// ── Create the rate limiter ─────────────────────────────────────
	//
	// Token bucket algorithm:
	//   - The bucket fills up at `rps` tokens per second
	//   - Each request costs 1 token
	//   - The bucket can hold at most `burst` tokens
	//   - If the bucket is empty, the request blocks until a token is available
	//
	// Example with rps=10, burst=20:
	//   - Steady state: 10 requests/second
	//   - If the server was idle for 2 seconds, 20 tokens accumulate
	//   - Those 20 can be used in a burst, then back to 10/sec
	//
	// rate.Limit(rps) converts float64 → rate.Limit type
	// rate.NewLimiter returns the limiter — call limiter.Wait(ctx) before each request

	limiter := rate.NewLimiter(rate.Limit(*rps), *burst)

	// ── Create channels for work distribution ───────────────────────
	//
	// Channels are Go's way of passing data between goroutines safely.
	// Think of them like a thread-safe queue.
	//
	// queryChan: main() puts queries in → workers take them out
	// resultChan: workers put results in → main() takes them out
	//
	// The number in make(chan, N) is the buffer size:
	//   - queryChan is unbuffered (0) — the sender blocks until a worker
	//     is ready to receive. This is fine because we want backpressure.
	//   - resultChan is buffered (len(queries)) — workers can send results
	//     without blocking, even if main() hasn't started collecting yet.

	queryChan := make(chan Query)
	resultChan := make(chan Result, len(queries))

	// ── Launch worker goroutines ────────────────────────────────────
	//
	// sync.WaitGroup tracks how many goroutines are still running.
	//   wg.Add(1) = "one more goroutine started"
	//   wg.Done() = "one goroutine finished"
	//   wg.Wait() = "block until all goroutines called Done()"
	//
	// We use it to know when ALL workers are done, so we can safely
	// collect results and write the output file.

	var wg sync.WaitGroup

	fmt.Printf("  Launching %d workers...\n", *workers)
	for i := 0; i < *workers; i++ {
		wg.Add(1)                    // tell WaitGroup: one more worker
		go worker(                   // launch a goroutine
			ctx,                     // context — for shutdown
			i,                       // worker ID (for logging)
			queryChan,               // where to get queries from
			resultChan,              // where to put results
			limiter,                 // rate limiter
			*vllmURL,                // vLLM server address
			*model,                  // model name
			&wg,                     // WaitGroup — call Done() when finished
		)
	}

	// ── Feed queries to workers ─────────────────────────────────────
	//
	// This goroutine sends each query into queryChan. Workers receive
	// from the other end. When all queries are sent, we close the
	// channel — this tells workers "no more work, finish up".
	//
	// We also check ctx.Done() to stop feeding if shutdown was requested.

	go func() {
		for _, q := range queries {
			select {
			case queryChan <- q:
				// query sent to a worker successfully
			case <-ctx.Done():
				// shutdown requested — stop sending queries
				fmt.Println("  Feed stopped — shutdown requested")
				close(queryChan)
				return
			}
		}
		close(queryChan) // all queries sent — tell workers there's no more work
	}()

	// ── Wait for all workers to finish ──────────────────────────────

	wg.Wait()          // blocks until all workers called wg.Done()
	close(resultChan)  // no more results will be sent

	// ── Collect results ─────────────────────────────────────────────
	//
	// range over a channel reads all values until the channel is closed.
	// Since we closed resultChan above, this loop will get every result
	// and then stop.

	var results []Result
	for r := range resultChan {
		results = append(results, r)
	}

	// ── Write results to JSON ───────────────────────────────────────

	if err := writeResults(*outputFile, results); err != nil {
		log.Fatalf("  Failed to write results: %v", err)
	}

	// ── Print summary ───────────────────────────────────────────────

	printSummary(results)
}

// ─────────────────────────────────────────────────────────────────────────────
// Worker function — runs in a goroutine, processes queries one at a time
// ─────────────────────────────────────────────────────────────────────────────

func worker(
	ctx context.Context,       // for shutdown detection
	id int,                     // worker ID (for logging)
	queries <-chan Query,       // receive-only channel: get queries from here
	results chan<- Result,      // send-only channel: put results here
	limiter *rate.Limiter,     // rate limiter: wait before each request
	vllmURL string,            // vLLM server base URL
	model string,              // model name to use
	wg *sync.WaitGroup,        // call Done() when this worker exits
) {
	defer wg.Done() // when this function returns, decrement the WaitGroup counter

	// range over a channel = keep receiving until the channel is closed.
	// When queryChan is closed (all queries sent), this loop exits and
	// the function returns (triggering defer wg.Done()).
	for q := range queries {

		// ── Check if shutdown was requested ──────────────────────
		// select with default is a non-blocking check.
		// If ctx.Done() has a value, shutdown was requested → stop.
		// If not, the default case runs → continue processing.
		select {
		case <-ctx.Done():
			return // stop this worker
		default:
			// continue processing
		}

		// ── Wait for rate limiter ───────────────────────────────
		//
		// limiter.Wait(ctx) blocks until a token is available OR
		// the context is cancelled. If the rate limit is 10 req/sec,
		// this ensures we never exceed that rate across ALL workers.
		//
		// This is the key concurrency control: 8 workers share one
		// rate limiter, so even though 8 goroutines run in parallel,
		// they collectively don't exceed the configured RPS.
		if err := limiter.Wait(ctx); err != nil {
			// context was cancelled while waiting → shutdown
			return
		}

		// ── Track active workers metric ─────────────────────────
		activeWorkers.Inc() // +1 active worker (Gauge goes up)

		// ── Execute the query ───────────────────────────────────
		result := executeQuery(ctx, q, vllmURL, model)

		// ── Record metrics ──────────────────────────────────────
		activeWorkers.Dec() // -1 active worker (Gauge goes down)

		framework := q.Framework
		if framework == "" {
			framework = "unknown"
		}

		// Record latency in the histogram
		queryDuration.WithLabelValues(framework).Observe(result.TotalMs / 1000.0)          // convert ms → seconds
		generationDuration.WithLabelValues(framework).Observe(result.GenerationMs / 1000.0)

		// Record success or error
		if result.Error != "" {
			queriesTotal.WithLabelValues(framework, "error").Inc()
			errorsTotal.WithLabelValues(framework).Inc()
		} else {
			queriesTotal.WithLabelValues(framework, "success").Inc()
		}

		// ── Send result back ────────────────────────────────────
		results <- result
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// executeQuery — sends one question to the vLLM server and returns the result
// ─────────────────────────────────────────────────────────────────────────────

func executeQuery(ctx context.Context, q Query, vllmURL string, model string) Result {
	result := Result{
		Question:    q.Question,
		GroundTruth: q.GroundTruth,
		Domain:      q.Domain,
		Framework:   q.Framework,
	}

	// ── Build the prompt ────────────────────────────────────────────
	//
	// We use the OpenAI chat completions format (system + user messages).
	// The system message sets the persona. The user message contains
	// the context (retrieved documents) and the question.
	//
	// If no context is provided (e.g., testing the orchestrator standalone),
	// we just ask the question directly.

	systemMsg := "You are a helpful assistant. Answer the question based only on the provided context. If the context does not contain the answer, say so."

	userMsg := q.Question
	if q.Context != "" {
		userMsg = fmt.Sprintf("Context:\n%s\n\nQuestion: %s", q.Context, q.Question)
	}

	chatReq := ChatRequest{
		Model: model,
		Messages: []ChatMessage{
			{Role: "system", Content: systemMsg},
			{Role: "user", Content: userMsg},
		},
	}

	// ── Serialize to JSON ───────────────────────────────────────────
	//
	// json.Marshal converts the Go struct into a JSON byte slice.
	// For example: {"model":"meta-llama/...", "messages":[...]}

	body, err := json.Marshal(chatReq)
	if err != nil {
		result.Error = fmt.Sprintf("marshal error: %v", err)
		return result
	}

	// ── Create the HTTP request ─────────────────────────────────────
	//
	// http.NewRequestWithContext creates a request tied to our context.
	// If the context is cancelled (Ctrl+C), the HTTP request is
	// automatically aborted — no zombie connections left behind.

	url := vllmURL + "/v1/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		result.Error = fmt.Sprintf("request creation error: %v", err)
		return result
	}
	req.Header.Set("Content-Type", "application/json")

	// ── Send the request and measure time ───────────────────────────
	//
	// time.Now() captures the current timestamp.
	// time.Since(start) calculates the duration.
	// We measure the full HTTP round-trip — this is the generation time
	// since vLLM does all the work server-side.

	start := time.Now()
	resp, err := http.DefaultClient.Do(req)
	generationMs := float64(time.Since(start).Milliseconds())

	if err != nil {
		result.Error = fmt.Sprintf("http error: %v", err)
		result.TotalMs = generationMs
		result.GenerationMs = generationMs
		return result
	}
	defer resp.Body.Close() // always close the response body to free resources

	// ── Read and parse the response ─────────────────────────────────

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		result.Error = fmt.Sprintf("read body error: %v", err)
		result.TotalMs = generationMs
		result.GenerationMs = generationMs
		return result
	}

	// Check HTTP status code — anything other than 200 is an error
	if resp.StatusCode != http.StatusOK {
		result.Error = fmt.Sprintf("vLLM returned status %d: %s", resp.StatusCode, string(respBody))
		result.TotalMs = generationMs
		result.GenerationMs = generationMs
		return result
	}

	// Parse the JSON response into our ChatResponse struct
	var chatResp ChatResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		result.Error = fmt.Sprintf("unmarshal error: %v", err)
		result.TotalMs = generationMs
		result.GenerationMs = generationMs
		return result
	}

	// Extract the answer text from the first choice
	if len(chatResp.Choices) > 0 {
		result.Answer = chatResp.Choices[0].Message.Content
	} else {
		result.Error = "no choices in response"
	}

	totalMs := float64(time.Since(start).Milliseconds())
	result.TotalMs = totalMs
	result.GenerationMs = generationMs

	return result
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions
// ─────────────────────────────────────────────────────────────────────────────

// loadQueries reads the input JSON file and returns a slice of Query structs.
func loadQueries(path string) ([]Query, error) {
	// os.ReadFile reads the entire file into memory as a byte slice.
	// For a file of 5,000 queries (maybe 5MB), this is fine.
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read file %s: %w", path, err)
	}

	// json.Unmarshal parses the JSON bytes into a Go slice of Query structs.
	// The JSON should be an array: [{"question": "...", ...}, ...]
	var queries []Query
	if err := json.Unmarshal(data, &queries); err != nil {
		return nil, fmt.Errorf("parse JSON: %w", err)
	}

	return queries, nil
}

// writeResults writes the results slice to a JSON file.
func writeResults(path string, results []Result) error {
	// json.MarshalIndent is like json.Marshal but adds indentation
	// for human readability. "" is the prefix, "  " is the indent.
	data, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal results: %w", err)
	}

	// os.WriteFile writes the byte slice to a file.
	// 0644 is the Unix permission: owner can read+write, others can read.
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("write file %s: %w", path, err)
	}

	fmt.Printf("  Results written to %s (%d entries)\n", path, len(results))
	return nil
}

// printSummary displays a quick overview of the benchmark results.
func printSummary(results []Result) {
	fmt.Println("\n═══════════════════════════════════════════════════════════")
	fmt.Println("  Summary")
	fmt.Println("═══════════════════════════════════════════════════════════")

	// Count results by framework
	// map[string]... is Go's dictionary type — like Python's dict.
	counts := make(map[string]int)
	errors := make(map[string]int)
	totalMs := make(map[string]float64)

	for _, r := range results {
		fw := r.Framework
		if fw == "" {
			fw = "unknown"
		}
		counts[fw]++
		if r.Error != "" {
			errors[fw]++
		}
		totalMs[fw] += r.TotalMs
	}

	for fw, count := range counts {
		avgMs := 0.0
		if count > 0 {
			avgMs = totalMs[fw] / float64(count)
		}
		errCount := errors[fw]
		fmt.Printf("  %-15s  queries: %4d  errors: %3d  avg latency: %8.1f ms\n",
			fw, count, errCount, avgMs)
	}

	fmt.Printf("\n  Total: %d queries processed\n", len(results))
	fmt.Println("═══════════════════════════════════════════════════════════")
}
