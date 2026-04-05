"""
tracing.py — Arize Phoenix + OpenTelemetry instrumentation

This module sets up distributed tracing for the RAG benchmark.
When enabled, every LLM call, retrieval, and evaluation step is
recorded as a "span" and sent to Phoenix for visual inspection.

HOW THIS WORKS (the 30-second version):
────────────────────────────────────────
1. OpenTelemetry (OTel) is a standard for structured logging ("traces").
2. A "trace" represents one complete operation (e.g., answering question #37).
3. A "span" is one step within a trace (e.g., "embed the question", "call GPT").
4. Spans nest: a "query" span contains a "retrieval" span and a "generation" span.
5. We export spans via OTLP (OTel's wire protocol) to Phoenix on port 4317.
6. Phoenix stores them and renders a UI at http://localhost:6006.

WHAT GETS TRACED AUTOMATICALLY:
────────────────────────────────
- LangChain: every chain.invoke(), retriever call, LLM call, embedding call
- LlamaIndex: every query_engine.query(), retriever.retrieve(), LLM call
- DSPy: every dspy.ChainOfThought() call, dspy.LM() call
- All three: token counts, model names, prompts, completions

WHAT WE ADD MANUALLY:
─────────────────────
- "benchmark_run" span wrapping the entire benchmark
- "framework_run" span per framework (langchain / llamaindex / dspy)
- "evaluation" span for the scoring phase (judge, RAGAS, BERTScore)

USAGE:
──────
    from src.evaluation.tracing import init_tracing, trace_span

    # Call once at startup — returns a cleanup function
    shutdown = init_tracing()

    # Use as a context manager to create manual spans
    with trace_span("my_operation", {"key": "value"}):
        do_something()

    # Call when done to flush all pending spans
    shutdown()

REQUIRES (pip install):
───────────────────────
    opentelemetry-api                      — the core OTel types (Tracer, Span, etc.)
    opentelemetry-sdk                      — the implementation (TracerProvider, exporters)
    opentelemetry-exporter-otlp-proto-grpc — sends spans to Phoenix over gRPC
    openinference-instrumentation-langchain — auto-traces LangChain operations
    openinference-instrumentation-llama-index — auto-traces LlamaIndex operations
    openinference-instrumentation-dspy     — auto-traces DSPy operations
"""

# ──────────────────────────────────────────────────────────────────────
# Imports — each one explained
# ──────────────────────────────────────────────────────────────────────

# contextmanager lets us write `with trace_span(...):` syntax
from contextlib import contextmanager

# ──────────────────────────────────────────────────────────────────────
# OpenTelemetry core types
# ──────────────────────────────────────────────────────────────────────

# `trace` is the top-level OTel module — gives us get_tracer() to create spans
from opentelemetry import trace

# TracerProvider is the "factory" that creates Tracer objects.
# Think of it like a database connection pool — you configure it once,
# then every part of your code can ask it for a Tracer.
from opentelemetry.sdk.trace import TracerProvider

# Resource describes WHERE the traces come from — it's metadata like
# "this is the rag-benchmark service". Phoenix uses this to label your
# traces in the UI so you can tell apart different applications.
from opentelemetry.sdk.resources import Resource

# SimpleSpanProcessor sends each span to the exporter immediately.
# There's also BatchSpanProcessor (batches them for efficiency), but
# Simple is easier to debug — you see spans in Phoenix right away.
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# OTLPSpanExporter sends spans over gRPC (a binary protocol, faster
# than HTTP/JSON) to whatever is listening on port 4317 — in our case,
# the Phoenix container from docker-compose.yml.
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# StatusCode lets us mark spans as OK or ERROR — Phoenix colors them
# green or red in the UI based on this.
from opentelemetry.trace import StatusCode


# ──────────────────────────────────────────────────────────────────────
# Module-level state
# ──────────────────────────────────────────────────────────────────────

# We store the tracer at module level so trace_span() can access it
# from anywhere without passing it around. None means tracing isn't
# initialized yet — trace_span() becomes a no-op.
_tracer = None


def init_tracing(
    phoenix_endpoint: str = "http://localhost:4317",
    service_name: str = "rag-benchmark",
) -> callable:
    """
    Initialize OpenTelemetry tracing and auto-instrument all three frameworks.

    Parameters
    ──────────
    phoenix_endpoint : str
        Where to send traces. Phoenix listens for OTLP gRPC on port 4317.
        This matches the docker-compose.yml port mapping:
            ports:
              - "4317:4317"   # OTLP gRPC trace ingestion

    service_name : str
        Label for this application in Phoenix UI. Shows up in the
        "Service" column so you can filter if multiple apps send traces.

    Returns
    ───────
    shutdown : callable
        Call this when the benchmark is done. It flushes any pending
        spans (makes sure nothing is lost) and cleans up resources.
        If you don't call this, the last few spans might not arrive
        at Phoenix because the process exits before they're sent.
    """

    # ── Step 1: Tell OTel "who we are" ──────────────────────────────
    #
    # Resource is a bag of key-value metadata. "service.name" is the
    # standard OTel attribute — Phoenix reads it to label traces.
    resource = Resource.create({
        "service.name": service_name,
    })

    # ── Step 2: Create the TracerProvider ────────────────────────────
    #
    # TracerProvider is the central OTel object. It:
    #   1. Creates Tracer objects (which create Spans)
    #   2. Holds configuration (Resource, SpanProcessors)
    #   3. Is registered globally so any library can find it
    provider = TracerProvider(resource=resource)

    # ── Step 3: Create the exporter ─────────────────────────────────
    #
    # The exporter converts Span objects into bytes and sends them
    # over the network. OTLPSpanExporter speaks gRPC (binary protocol).
    #
    # `insecure=True` means no TLS encryption. This is fine because
    # both the benchmark and Phoenix run on the same machine (localhost).
    # In production you'd use TLS, but for local development it adds
    # complexity for zero security benefit.
    exporter = OTLPSpanExporter(
        endpoint=phoenix_endpoint,
        insecure=True,   # no TLS — both processes are on localhost
    )

    # ── Step 4: Wire the exporter to the provider ───────────────────
    #
    # SpanProcessor is the middleman between "span is created" and
    # "span is sent to Phoenix". SimpleSpanProcessor = send immediately.
    # BatchSpanProcessor would buffer spans and send in bulk (more
    # efficient but adds latency — not worth it for a benchmark).
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # ── Step 5: Register as the global TracerProvider ───────────────
    #
    # This is the key line. After this, ANY code that calls
    # `trace.get_tracer(...)` — including the auto-instrumentors —
    # gets a Tracer backed by our provider, which sends to Phoenix.
    # Without this, OTel uses a no-op provider that discards everything.
    trace.set_tracer_provider(provider)

    # ── Step 6: Create our own tracer for manual spans ──────────────
    #
    # get_tracer() takes a name (any string) that shows up in Phoenix
    # as the "instrumentation library" — helps you tell apart spans
    # from your code vs spans from LangChain auto-instrumentation.
    global _tracer
    _tracer = trace.get_tracer("rag-benchmark")

    # ── Step 7: Auto-instrument the three frameworks ────────────────
    #
    # Each instrumentor monkey-patches (modifies at runtime) the
    # framework's internal functions to emit OTel spans. For example,
    # the LangChain instrumentor wraps chain.invoke() so every call
    # automatically creates a span with the prompt, response, and
    # token counts — no changes to langchain_rag/pipeline.py needed.
    #
    # We wrap each in try/except because the auto-instrumentors are
    # optional dependencies — the benchmark should still work without
    # them, just without automatic tracing.

    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor

        # instrument() patches LangChain globally. All LangChain
        # operations (chain.invoke, retriever.invoke, llm.invoke)
        # will now emit spans to Phoenix.
        LangChainInstrumentor().instrument()
        print("  Tracing: LangChain auto-instrumented")
    except ImportError:
        # If openinference-instrumentation-langchain isn't installed,
        # LangChain still works — just without automatic trace spans.
        print("  Tracing: LangChain instrumentor not installed (pip install openinference-instrumentation-langchain)")

    try:
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

        # Same idea — patches LlamaIndex's query engines, retrievers,
        # and LLM calls to emit spans.
        LlamaIndexInstrumentor().instrument()
        print("  Tracing: LlamaIndex auto-instrumented")
    except ImportError:
        print("  Tracing: LlamaIndex instrumentor not installed (pip install openinference-instrumentation-llama-index)")

    try:
        from openinference.instrumentation.dspy import DSPyInstrumentor

        # Patches dspy.LM() calls and dspy.Module.forward() calls.
        DSPyInstrumentor().instrument()
        print("  Tracing: DSPy auto-instrumented")
    except ImportError:
        print("  Tracing: DSPy instrumentor not installed (pip install openinference-instrumentation-dspy)")

    print(f"  Tracing: initialized → sending to {phoenix_endpoint}")
    print(f"  Tracing: open http://localhost:6006 to view traces")

    # ── Step 8: Return a shutdown function ──────────────────────────
    #
    # provider.shutdown() does two things:
    #   1. Flushes: sends any spans still in memory to Phoenix
    #   2. Cleans up: closes gRPC connections, frees resources
    # The caller should call this when the benchmark is done.
    def shutdown():
        provider.shutdown()
        print("  Tracing: shut down (all spans flushed)")

    return shutdown


# ──────────────────────────────────────────────────────────────────────
# Manual span helper
# ──────────────────────────────────────────────────────────────────────

@contextmanager
def trace_span(name: str, attributes: dict = None):
    """
    Context manager that creates a manual OTel span.

    Usage:
        with trace_span("evaluate_langchain", {"framework": "langchain", "n_pairs": 50}):
            results = run_framework(...)
            scores = evaluate_all(...)

    What this does:
    1. Creates a new span named "evaluate_langchain"
    2. Attaches key-value attributes (visible in Phoenix UI)
    3. Records the start time
    4. Runs your code inside the `with` block
    5. If your code raises an exception:
       - Records the error on the span (shows up red in Phoenix)
       - Re-raises the exception (your code still sees the error)
    6. Records the end time
    7. Sends the span to Phoenix

    The span automatically nests inside any parent span. If you do:
        with trace_span("benchmark"):         # parent
            with trace_span("langchain"):     # child
                ...
    Phoenix shows "langchain" as a child of "benchmark" in the trace tree.

    Parameters
    ──────────
    name : str
        Span name — shows up as the label in Phoenix's trace waterfall.
        Use descriptive names like "evaluate_langchain" not "step_2".

    attributes : dict, optional
        Key-value pairs attached to the span. Phoenix shows these in
        the span detail panel. Use them for anything that helps debugging:
        framework name, number of queries, model name, etc.
        Keys must be strings. Values can be strings, ints, floats, or bools.
    """

    # If init_tracing() was never called, _tracer is None.
    # In that case, do nothing — just run the code block without tracing.
    # This means run_benchmark.py works identically with or without --trace.
    if _tracer is None:
        yield    # yield = "run the code inside the `with` block"
        return   # then exit without doing anything else

    # start_as_current_span() does three things:
    #   1. Creates a new span with the given name
    #   2. Sets it as the "current" span (so child spans know their parent)
    #   3. Returns it via `as span` so we can add attributes/errors
    with _tracer.start_as_current_span(name) as span:

        # Attach the attributes to the span.
        # These are visible in Phoenix when you click on the span.
        if attributes:
            for key, value in attributes.items():
                # set_attribute() only accepts primitive types:
                # str, int, float, bool, or sequences of those.
                # We str() everything to be safe — a dict or list
                # would cause OTel to silently drop the attribute.
                span.set_attribute(key, str(value) if not isinstance(value, (str, int, float, bool)) else value)

        try:
            yield span   # run the caller's code block, giving them the span object
        except Exception as e:
            # record_exception() stores the error type, message, and
            # stack trace on the span. Phoenix renders this as a red
            # error badge with an expandable traceback.
            span.record_exception(e)

            # set_status(ERROR) marks the span as failed. Phoenix
            # uses this to color the span red and filter for errors.
            span.set_status(StatusCode.ERROR, str(e))

            # Re-raise so the caller's error handling still works.
            # We're just observing the error, not swallowing it.
            raise


def add_span_event(name: str, attributes: dict = None):
    """
    Add a point-in-time event to the current span.

    Events are like log lines attached to a span. Use them for
    milestones within a long operation:

        with trace_span("evaluate_langchain"):
            run_string_overlap()
            add_span_event("string_overlap_done")
            run_bertscore()
            add_span_event("bertscore_done", {"f1": 0.82})
            run_judge()

    In Phoenix, events show up as dots on the span's timeline bar.

    Parameters
    ──────────
    name : str
        Event name (e.g., "bertscore_done").
    attributes : dict, optional
        Key-value pairs. Same rules as trace_span attributes.
    """
    # Get the currently-active span (set by the nearest trace_span context manager)
    current_span = trace.get_current_span()

    # INVALID_SPAN is a sentinel returned when there's no active span
    # (either tracing is off or we're not inside a trace_span block).
    if current_span is None or not current_span.is_recording():
        return

    # Convert all values to OTel-safe types
    safe_attrs = {}
    if attributes:
        for k, v in attributes.items():
            safe_attrs[k] = str(v) if not isinstance(v, (str, int, float, bool)) else v

    current_span.add_event(name, attributes=safe_attrs)
