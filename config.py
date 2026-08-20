"""config.py — centralised settings loaded from .env"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Ollama
    ollama_inference_url: str = "http://localhost:11434"
    ollama_embed_url: str = "http://localhost:11434"
    ollama_inference_model: str = "gemma4:31b-cloud"
    ollama_inference_critic_model: str = "gpt-oss:120b-cloud"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_image_processing_model: str = "gemma4:31b-cloud"
    ollama_inference_api_key: str = ""   # blank = no auth (default self-hosted Ollama)
    ollama_embed_api_key: str = ""       # blank = no auth; set if embed endpoint is gated
    ollama_num_ctx: int = 200000

    # ------------------------------------------------------------------
    # Latency tuning
    # ------------------------------------------------------------------
    # How long Ollama keeps a model resident in VRAM after a request. The
    # default is 5 minutes, so a thread picked up 6 minutes later pays a full
    # cold model load before the first token — routinely the single largest
    # component of time-to-first-token on a local setup. "30m" keeps the model
    # warm across a normal working session; "-1" pins it indefinitely (watch
    # VRAM if you run other models alongside).
    ollama_keep_alive: str = "30m"

    # The critic only ever reads the answer plus context already truncated to
    # 50k chars in agents/critic.py, and emits a small JSON verdict. Allocating
    # the full 200k KV cache for that is wasted setup time on every turn, so it
    # gets its own smaller window. Raise it only if you also raise
    # MAX_CONTEXT_CHARS in agents/critic.py.
    ollama_critic_num_ctx: int = 32000

    # RAG thresholds
    rag_small_threshold: int = 2000
    rag_medium_threshold: int = 15000

    # Chunking
    chunk_breakpoint_type: str = "percentile"
    chunk_breakpoint_threshold: int = 95
    chunk_min_size: int = 150
    chunk_max_size: int = 800

    # SQLite
    db_path: str = "data/webpulse.db"
    embed_dimensions: int = 768

    # Scraper
    scraper_timeout: int = 28
    scraper_max_headlines: int = 50
    playwright_wait_seconds: float = 4.0
    playwright_headless: bool = True

    # Deferred scrape escalation (background Playwright renders)
    scrape_defer_enabled: bool = True        # false = old blocking behaviour
    scrape_fast_deadline: float = 2.5        # secs before cheap tiers hand off
    scrape_max_deferred: int = 6             # max background renders per run
    scrape_max_concurrent_heavy: int = 3     # concurrent Chromium renders
    scrape_late_wait_seconds: float = 5.0   # bounded wait before final answer

    # Morning brief batch
    brief_auto_run: bool = True          # set false to disable auto morning run
    brief_start_time: str = "08:15"
    brief_articles_per_site: int = 10
    brief_timezone: str = "Asia/Kolkata"

    # SMTP
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_to: str = ""
    smtp_enabled: bool = False
    portfolio_symbols: str = ""              # fallback if DB portfolio empty e.g. "RELIANCE,TCS"
    portfolio_analysis_enabled: bool = True  # set false to skip per-holding Q&A in brief
    brief_prompts_json: str = ""             # JSON override for DEFAULT_PROMPTS; empty = use defaults
    brief_default_category: str = "Markets"  # category name for auto-run scheduler

    # URL Queue / Background crawler
    crawl_queue_enabled: bool = True         # enable background crawl queue refresh
    crawl_queue_interval_mins: int = 45      # how often to refresh (minutes)
    crawl_queue_start_hour: int = 6          # start crawling at this hour (local tz)
    crawl_queue_end_hour: int = 21           # stop crawling after this hour

    # Embedding provider
    embed_provider: str = "ollama"          # "ollama" | "google" | "openai" | "cohere"
    google_api_key: str = ""                # Google Generative AI API key
    google_embed_model: str = "gemini-embedding-001"  # recommended, replaces text-embedding-004
    openai_api_key: str = ""
    openai_embed_model: str = "text-embedding-3-small"
    cohere_api_key: str = ""
    cohere_embed_model: str = "embed-english-v3.0"
    jina_api_key: str = ""
    jina_embed_model: str = "jina-embeddings-v3"  # 65.5 MTEB, 1024 dims, free tier

    # Crawler settings
    crawl_window_hours: int = 4          # how far back to look for recent pages
    crawl_max_results: int = 50          # max URLs fetched from sitemap/RSS before filtering
    crawl_top_n: int = 15               # how many to scrape after relevance filtering
    crawl_min_score: float = 0.70       # minimum embedding similarity score to include

    @property
    def brief_default_prompts(self) -> list[dict]:
        """Default insight prompts for morning brief. Override via BRIEF_PROMPTS_JSON."""
        return [
            {
                "key": "trending_news",
                "label": "📰 Top Trending News",
                "prompt": (
                    "What are the top 10 trending news stories today that may impact market dynamics? "
                    "List them as an HTML numbered list with a one-line explanation of potential market impact for each."
                ),
            },
            {
                "key": "market_outlook",
                "label": "📈 India Market Outlook",
                "prompt": (
                    "Based on today's news context, how is the Indian stock market (Sensex/Nifty) likely to behave today? "
                    "Consider global cues, FII/DII activity, sector trends, and macro factors. "
                    "Give a clear directional view: bullish, bearish, or range-bound, with key reasons."
                ),
            },
            {
                "key": "stock_calls",
                "label": "🎯 Expert Stock Recommendations",
                "prompt": (
                    "Based on the news context, which specific stocks have been explicitly recommended by analysts or experts? "
                    "Create an HTML table with columns: Stock, Recommendation (BUY/SELL/HOLD), Target Price (if mentioned), "
                    "Analyst/Source, and Key Reason. Only include stocks with explicit recommendations in the news."
                ),
            },
            {
                "key": "focus_areas",
                "label": "🔍 Focus Areas Today",
                "prompt": (
                    "Based on today's news, what are the key focus areas, themes, or sectors that investors should watch today? "
                    "Include: sectors in spotlight, key events or data releases, geopolitical factors, and any earnings announcements. "
                    "Format as an HTML table with columns: Area, Why It Matters, Likely Impact."
                ),
            },
            {
                "key": "risk_factors",
                "label": "⚠️ Risk Factors & Caution Zones",
                "prompt": (
                    "Based on today's news context, what are the key risk factors or caution zones for the market today? "
                    "Include global risks, domestic concerns, overvalued sectors, or stocks facing headwinds. "
                    "Be specific and factual. Format as a concise HTML bulleted list."
                ),
            },
        ]

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 1976
    app_debug: bool = True
    app_title: str = "Web Pulse"

    # ------------------------------------------------------------------
    # Chat persistence (conversations + messages)
    # ------------------------------------------------------------------
    chat_db_path: str = "data/chat.db"
    chat_history_turns: int = 6           # turns of history handed to the supervisor

    # ------------------------------------------------------------------
    # Uploads
    # ------------------------------------------------------------------
    upload_dir: str = "data/uploads"
    upload_max_mb: int = 25
    upload_max_files: int = 10
    ingest_concurrency: int = 2           # documents ingested in parallel

    # ------------------------------------------------------------------
    # Document chunking — bridges to the reference ChunkerConfig.
    # NOTE: chunk_min_size above (150) is the *semantic* chunker's floor from
    # the original RAG settings; the prose chunker used for uploads wants a
    # much smaller merge threshold, hence chunk_min_chars being separate.
    # ------------------------------------------------------------------
    chunk_target_chars: int = 1200
    chunk_max_chars: int = 2400
    chunk_overlap_chars: int = 150
    chunk_min_chars: int = 80             # chunks below this merge with the next

    # ------------------------------------------------------------------
    # Enrichment / embedding of uploaded documents
    # ------------------------------------------------------------------
    enrich_mode: str = "auto"             # auto | spacy | none
    embed_batch_size: int = 16
    embed_concurrency: int = 4
    embed_timeout_seconds: int = 120
    classify_rules_path: str = ""         # optional YAML, owned by Security

    # ------------------------------------------------------------------
    # Retrieval over uploaded documents
    # ------------------------------------------------------------------
    rag_top_k: int = 6
    # How strongly retrieval prefers the more recently modified source when
    # candidates are otherwise close — 0.15 = up to a 15% score nudge, only
    # enough to flip near-ties, not override a clearly better semantic match
    # from an older document. Set to 0 to disable and restore pure relevance
    # ranking.
    rag_recency_weight: float = 0.15
    # Max chunks any single document can contribute to one answer's context.
    # Without this, several near-duplicate chunks from ONE version (old or
    # new) can fill the whole context window and leave no room for the other
    # version to be seen and compared at all.
    rag_max_chunks_per_doc: int = 3

    # ---------------- knowledge graph (v0: entity co-occurrence) ----------------
    # Points at your existing offline Neo4j instance. Everything graph-related
    # fails open if this is wrong or Neo4j is down — no graph, retrieval works
    # exactly as it does today, nothing else in the app is affected.
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j"
    neo4j_database: str = "neo4j"
    # Two mentions with compatible-but-not-identical names (e.g. "Rachpal
    # Singh" and "R Singh") only merge into one graph node when their
    # combined evidence score clears this bar — see docstore/entity_resolver.py
    # for what counts as evidence. Higher = more conservative (more distinct
    # nodes, fewer false merges); lower = more aggressive merging.
    graph_merge_threshold: float = 0.62
    # Facts pulled into the LLM's TEXT context per collection — words in a
    # prompt tolerate more volume than a diagram does.
    graph_hydrate_facts_per_collection: int = 12
    # Total nodes shown in the ON-DEMAND VISUAL trace, across ALL collections
    # combined, prioritized by edge weight. A separate, tighter number from
    # the one above on purpose: readable as a diagram is a much lower bar
    # than useful as words an LLM can skim.
    graph_trace_max_nodes: int = 15
    # Hard cap on how long graph hydration is allowed to add to a query's
    # latency. Every Neo4j call in docstore/graph_store.py is a blocking
    # synchronous call — first-connection especially, since verify_connectivity()
    # is a real network round trip. Without this bound, a slow or unreachable
    # Neo4j (a misconfigured database name, a firewall, the server being down)
    # freezes the request indefinitely — not gracefully degraded, genuinely
    # stuck. On timeout, hydration is treated exactly like "no graph exists":
    # skipped, logged, the main answer proceeds unaffected.
    graph_hydrate_timeout_s: float = 2.5

    # ---------------- observability (self-hosted Arize Phoenix) ----------------
    # Default OFF — see observability.py for the full rationale. Started with
    # LangSmith; switched to Phoenix because LangSmith's self-hosting is an
    # Enterprise-licensed add-on, discovered only after building against it.
    # Phoenix is MIT-licensed, genuinely self-hostable, and OpenTelemetry-native
    # — the actual vendor-neutral industry standard, not one vendor's format.
    #
    # NOTE the package name carefully if you ever touch this: `arize-otel`
    # (no "phoenix") points at Arize's separate COMMERCIAL cloud platform and
    # needs a paid space_id/api_key. `arize-phoenix-otel` is the open-source,
    # self-hosted one this app actually uses. Easy to grab the wrong one.
    phoenix_tracing_enabled: bool = False
    phoenix_endpoint: str = "http://localhost:6006"   # Phoenix server's default port
    phoenix_api_key: str = ""    # only needed if your instance has auth enabled
    phoenix_project: str = "chat-app"
    # Points at Phoenix's general UI, not a guessed per-conversation deep
    # link — I could not confirm Phoenix's exact session-deep-link URL
    # structure from documentation, and guessing wrong here (as happened
    # with the LangSmith URL template) produces a dead link that looks like
    # it should work. Safer to land on the UI and let the person filter by
    # session_id (== conversation_id) themselves, using Phoenix's own filter
    # UI, until the real deep-link path is confirmed against a live instance.
    phoenix_trace_url_template: str = "{endpoint}"
    # Auto-launch `phoenix serve` as a genuine SEPARATE OS process (not
    # embedded — Arize's own guidance is that in-process launch is a
    # notebook-only mode and "may not work as expected" in a real app) when
    # tracing is enabled and nothing is already listening at phoenix_endpoint.
    # Set False if Phoenix is managed independently (systemd, Docker, its
    # own supervisor) and this app should only ever connect to it, never
    # start it.
    phoenix_auto_launch: bool = True
    # Graph building runs as its OWN opt-in job kind, deliberately separate
    # from ingestion — embedding already takes real time, and nobody should
    # have graph construction silently tacked onto a folder-ingestion run
    # they didn't ask to be slower.
    graph_build_workers: int = 0   # 0 = auto, same rule as ingestion jobs

    # ---------------- critic loop ----------------
    # Below this score (0-100), route_from_critic() sends the response back
    # to the supervisor for another pass instead of returning it. Was a bare
    # constant in workflow/routing.py; moved here so it's one setting instead
    # of a value buried in routing logic, and so the critic's own prompt
    # (agents/critic.py) can reference the SAME number rather than an
    # independently hardcoded one that could silently drift out of sync.
    critic_pass_threshold: int = 45
    # A retry that isn't improving is wasted latency, not diligence. If the
    # score gained less than this many points over the PREVIOUS attempt, the
    # loop stops even below critic_pass_threshold rather than spending
    # another full regeneration + evaluation cycle on a plateaued answer.
    critic_min_improvement: int = 5
    # Bounded separately from the critic loop count on purpose: "how many
    # times should the answer be regenerated" and "how many times should we
    # re-query the same corpus hoping for a different result" are different
    # questions. A corpus that didn't have the answer the first time won't
    # have it the third — retrying retrieval past this just burns loops.
    max_retrieval_attempts: int = 2
    # General ceiling on how many times ANY single agent can be dispatched
    # within one turn. Distinct from max_loops (which bounds supervisor
    # iterations) — a supervisor that keeps re-dispatching the same worker
    # every loop, each time producing another artifact, burns loops and
    # confuses the answer. Observed with mermaid_generator: four diagrams
    # generated across four loops, all four presented as the answer, when
    # the first had already succeeded.
    max_dispatches_per_agent: int = 2
    # Width budget for agents declared "fanout": True in AGENT_REGISTRY
    # (workflow/registry.py) — payload-determined agents like scraper/search/
    # doc_retriever, where dispatching N times in a turn means N different
    # targets, not the same work repeated. See registry.py's docstring for
    # how a new agent opts into this instead of the tighter cap above.
    max_fanout_per_agent: int = 8

    # ---------------- batch ingestion jobs ----------------
    # Folders outside these roots are refused by the jobs API. Without the
    # confinement the folder textbox on the job form is an arbitrary-file-read
    # endpoint. os.pathsep separated (":" on Linux, ";" on Windows).
    ingest_allowed_roots: str = "data/incoming"
    job_staging_dir: str = "data/staging"
    job_default_workers: int = 0          # 0 = cores minus 25% headroom
    job_embed_concurrency: int = 4        # network-bound; independent of cores
    job_vision_concurrency: int = 2       # image extraction, cloud rate-limited

    # ---------------- code chunking ----------------
    # Wider than the prose windows: code is denser, and a function cut in half
    # retrieves as confidently-cited nonsense.
    code_chunk_target_chars: int = 1600
    code_chunk_max_chars: int = 3200
    code_chunk_min_chars: int = 120
    code_chunk_preamble_chars: int = 400  # imports carried into later chunks
    rag_context_max_chars: int = 12000


@lru_cache
def get_settings() -> Settings:
    return Settings()