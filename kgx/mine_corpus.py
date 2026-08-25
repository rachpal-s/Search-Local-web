"""kgx/mine_corpus.py — READ-ONLY corpus analysis, v2.

Writes nothing to the database.

WHY v2 EXISTS
-------------
v1 aggregated doc_chunks.entities and proved that column unusable: every
mention in the corpus carries the label PROPN, which only the no-spaCy
fallback path emits, and that path's regex cannot match an acronym at all.
In a Cloud/AI corpus that excludes AWS, API, RAG, MCP, LLM, IAM, S3 — most
of the actual vocabulary. So `terms` below ignores the entities column
entirely and mines the raw chunk TEXT.

v1's duplicate pass also over-merged, via connected-components transitivity:
one cluster absorbed 30% of the corpus because template boilerplate shared
between otherwise unrelated documents chained them together. `dupes` below
suppresses boilerplate, scores symmetrically, links with complete-linkage
rather than any-link, and separates CONTAINMENT (a later cumulative document
absorbing an earlier one) from NEAR-DUPLICATE (two renderings of one thing).

    python -m kgx.mine_corpus terms
    python -m kgx.mine_corpus terms --min-docs 3 --top 600
    python -m kgx.mine_corpus dupes
    python -m kgx.mine_corpus dupes --jaccard 0.55 --containment 0.85
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from docstore import store

OUT_DIR = Path("kgx/out")

# --------------------------------------------------------------------------
# SEED GAZETTEER — the deterministic extraction route (framework §7).
# Deliberately partial. Its job is not to be complete on day one; it is to
# (a) prove the routing works and (b) let the `terms` pass report what it
# does NOT yet know, which is the triage list that grows this table.
# Format: canonical -> (class, vendor_or_owner)
# --------------------------------------------------------------------------
GAZETTEER: dict[str, tuple[str, str]] = {}


def _seed(cls: str, owner: str, names: str) -> None:
    for n in names.split(","):
        n = n.strip()
        if n:
            GAZETTEER[n] = (cls, owner)


_seed("Vendor", "", "AWS,Amazon Web Services,Microsoft Azure,Azure,Google Cloud,GCP,"
      "OpenAI,Anthropic,Databricks,Snowflake,HashiCorp,Confluent,Neo4j,Elastic,"
      "MongoDB,Red Hat,VMware,IBM,Oracle,Salesforce,ServiceNow,NVIDIA,Meta,Mistral,"
      "Cohere,Hugging Face,Pinecone,Qdrant,Weaviate,Datadog,Splunk,Grafana Labs")

_seed("Service", "AWS", "Amazon Bedrock,Bedrock,AWS Lambda,Amazon S3,Amazon EKS,Amazon ECS,"
      "Amazon SageMaker,SageMaker,Amazon RDS,DynamoDB,Amazon Kendra,AWS Fargate,"
      "AWS Step Functions,Amazon EventBridge,AWS Glue,Amazon Neptune,AWS IAM,"
      "Amazon CloudWatch,AWS CloudFormation,Amazon API Gateway,AWS Transit Gateway,"
      "Amazon OpenSearch,AWS Control Tower,AWS Organizations,Amazon Q,AWS PrivateLink")

_seed("Service", "Azure", "Azure OpenAI,Azure OpenAI Service,Azure AI Foundry,Azure AI Search,"
      "Azure Functions,Azure Kubernetes Service,AKS,Azure Monitor,Azure Policy,"
      "Azure Key Vault,Azure Data Factory,Azure Synapse,Azure Cosmos DB,Azure DevOps,"
      "Azure API Management,Azure Front Door,Azure Entra ID,Entra ID,Azure Blob Storage,"
      "Azure Container Apps,Azure Landing Zone,Azure Machine Learning,Azure Databricks")

_seed("Service", "GCP", "Vertex AI,Google Kubernetes Engine,GKE,BigQuery,Cloud Run,"
      "Cloud Functions,Pub/Sub,Cloud Spanner,Dataflow,Cloud Storage,Gemini API,"
      "Document AI,Cloud Armor,Anthos,Dataplex,Looker")

_seed("Model", "", "GPT-4,GPT-4o,GPT-5,Claude,Claude Opus,Claude Sonnet,Claude Haiku,"
      "Gemini,Gemini Pro,Llama,Llama 3,Mistral 7B,Mixtral,Titan,Nova,Phi-3,DeepSeek,"
      "Qwen,Command R,Embed v3,text-embedding-3-small,nomic-embed-text,BERT,T5")

_seed("Framework", "", "LangChain,LangGraph,LlamaIndex,Semantic Kernel,AutoGen,CrewAI,"
      "Haystack,DSPy,Instructor,Outlines,Guidance,Ragas,DeepEval,PydanticAI,"
      "Strands,Bedrock Agents,OpenAI Agents SDK,Google ADK,Temporal,Airflow,Dagster,"
      "Prefect,Ray,Spark,Apache Spark,Apache Kafka,Kafka,Flink,dbt,Great Expectations")

_seed("Library", "python", "FastAPI,Pydantic,SQLAlchemy,Jinja2,Uvicorn,Celery,pandas,"
      "NumPy,scikit-learn,PyTorch,TensorFlow,spaCy,GLiNER,transformers,sentence-transformers,"
      "FAISS,hnswlib,RapidFuzz,Splink,pySHACL,LinkML,rdflib,httpx,Streamlit,Gradio,pytest")

_seed("Platform", "", "Kubernetes,K8s,Docker,Terraform,Pulumi,Helm,Argo CD,ArgoCD,"
      "Flux,Istio,Linkerd,Envoy,NGINX,Kong,Consul,Vault,Backstage,Crossplane,"
      "GitHub Actions,GitLab CI,Jenkins,OpenShift,Knative,KEDA,Prometheus,Grafana,"
      "Jaeger,Loki,Tempo,OpenTelemetry,OTel,Phoenix,LangFuse,Ollama,vLLM,Triton")

_seed("Protocol", "", "MCP,Model Context Protocol,A2A,Agent2Agent,OAuth,OAuth 2.0,OIDC,"
      "SAML,gRPC,GraphQL,REST,WebSocket,SSE,Server-Sent Events,AMQP,MQTT,SPARQL,"
      "Cypher,OpenAPI,AsyncAPI,JSON-RPC,SCIM,JWT,mTLS,SPIFFE,CloudEvents")

_seed("Pattern", "", "RAG,Retrieval Augmented Generation,GraphRAG,Agentic RAG,ReAct,"
      "Reflection,Chain of Thought,Tree of Thoughts,Multi-Agent,Supervisor Pattern,"
      "Human in the Loop,Guardrails,Zero Trust,Landing Zone,Service Mesh,Sidecar,"
      "Strangler Fig,Event Driven Architecture,CQRS,Event Sourcing,Saga,Circuit Breaker,"
      "Hub and Spoke,Well Architected,Twelve Factor,Microservices,Serverless,"
      "Blue Green Deployment,Canary Release,GitOps,Infrastructure as Code,Data Mesh,"
      "Medallion Architecture,Lakehouse,Feature Store,Semantic Layer,Knowledge Graph")

_seed("Capability", "", "Observability,Distributed Tracing,Entity Resolution,Vector Search,"
      "Hybrid Search,Reranking,Chunking,Embedding,Fine Tuning,Prompt Engineering,"
      "Evaluation,Red Teaming,Drift Detection,Lineage,Data Governance,Cost Allocation,"
      "Autoscaling,Multi Tenancy,Disaster Recovery,Chaos Engineering,FinOps,Platform Engineering")

_seed("Regulation", "", "GDPR,DPDP,DPDPA,HIPAA,PCI DSS,SOC 2,ISO 27001,ISO 42001,"
      "NIST AI RMF,EU AI Act,AI Act,OWASP LLM Top 10,MITRE ATLAS,CIS Benchmark,"
      "FedRAMP,DORA,SOX,CCPA,RBI,SEBI")

_seed("Risk", "", "Prompt Injection,Jailbreak,Data Exfiltration,Hallucination,"
      "Model Poisoning,Supply Chain Attack,Privilege Escalation,Shadow AI,"
      "Tool Misuse,Excessive Agency,Sensitive Data Disclosure,Insecure Output Handling")

# Acronyms that must match case-sensitively — lowercasing them collides with
# ordinary English words ("IAM"/"I am", "ACT", "SAGA", "REST", "ACID").
CASE_SENSITIVE = {t for t in GAZETTEER if t.isupper() and len(t) <= 6}

# Heading/boilerplate words that dominate the fallback extractor's output and
# must never reach a candidate list.
_STOPFORMS = set("""the this that these those what how when where why who which
text image tables table figure note example summary overview introduction
conclusion appendix section chapter step phase part page source reference
key point points use uses using used all any each every one two three first
second next then also for and but with from into your you they them their
there here now new old high low non multi cross real time date name type
given however instead further overall finally therefore
question answer questions answers yes no true false
""".split())


def _scopes(collection: str | None) -> list[tuple[str, str]]:
    with store.conn() as c:
        if collection:
            row = c.execute(
                "SELECT id, title FROM conversations WHERE id=? OR title=?",
                (collection, collection)).fetchone()
            if not row:
                sys.exit(f"No conversation/collection matching {collection!r}")
            return [(row["id"], row["title"])]
        rows = c.execute("SELECT id, title FROM conversations WHERE kind='collection' "
                         "ORDER BY title").fetchall()
    if not rows:
        sys.exit("No collections found. Pass --collection <id>.")
    return [(r["id"], r["title"]) for r in rows]


def _write_tsv(path: Path, rows: list[dict], cols: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")).replace("\t", " ").replace("\n", " ")
                              for c in cols) + "\n")


# ------------------------------------------------------------------- terms

# Candidate shapes that the fallback regex structurally could not see.
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,7}(?:-[A-Z0-9]{1,6})?\b")
_CAMEL = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b")             # LangGraph, DynamoDB
_DOTTED = re.compile(r"\b[a-z][a-z0-9_]*(?:[.-][a-z0-9_]+){1,3}\b")  # scikit-learn
_TITLE_SEQ = re.compile(r"\b[A-Z][a-zA-Z0-9]{2,}(?:\s+[A-Z][a-zA-Z0-9]{2,}){1,3}\b")
_VERSIONED = re.compile(r"\b[A-Z][A-Za-z]+\s?\d+(?:\.\d+)?\b")       # GPT 4, Llama 3


def mine_terms(scope_ids: list[str], min_docs: int, top: int) -> None:
    """Match the seed gazetteer against raw chunk text, and rank everything it
    did NOT recognise as the triage queue for growing the gazetteer."""
    placeholders = ",".join("?" * len(scope_ids))
    with store.conn() as c:
        rows = c.execute(
            f"SELECT doc_id, text FROM doc_chunks "
            f"WHERE conversation_id IN ({placeholders})", scope_ids).fetchall()

    # Longest-first so "Azure OpenAI Service" wins over "Azure".
    ci_terms = sorted((t for t in GAZETTEER if t not in CASE_SENSITIVE),
                      key=len, reverse=True)
    ci_pat = re.compile(r"(?<![\w-])(" + "|".join(re.escape(t) for t in ci_terms)
                        + r")(?![\w-])", re.I)
    cs_pat = re.compile(r"(?<![\w-])(" + "|".join(re.escape(t) for t in CASE_SENSITIVE)
                        + r")(?![\w-])")

    hits: dict[str, dict] = {}
    unknown: dict[str, dict] = {}
    lookup = {t.lower(): t for t in GAZETTEER}
    known_lower = set(lookup)
    chunks_with_hits = 0
    multi_hit_chunks = 0

    for r in rows:
        text = r["text"] or ""
        found: set[str] = set()
        for m in ci_pat.finditer(text):
            found.add(lookup[m.group(1).lower()])
        for m in cs_pat.finditer(text):
            found.add(m.group(1))
        for canon in found:
            rec = hits.setdefault(canon, {"docs": set(), "chunks": 0})
            rec["docs"].add(r["doc_id"])
            rec["chunks"] += 1
        if found:
            chunks_with_hits += 1
        if len(found) >= 2:
            multi_hit_chunks += 1

        # Unrecognised candidates — the growth queue.
        cands: set[str] = set()
        for pat in (_ACRONYM, _CAMEL, _DOTTED, _TITLE_SEQ, _VERSIONED):
            for m in pat.finditer(text):
                s = m.group(0).strip()
                low = s.lower()
                if low in known_lower or low in _STOPFORMS or len(s) < 3:
                    continue
                if all(w in _STOPFORMS for w in low.split()):
                    continue
                cands.add(s)
        for s in cands:
            rec = unknown.setdefault(s, {"docs": set(), "chunks": 0})
            rec["docs"].add(r["doc_id"])
            rec["chunks"] += 1

    n_docs = len({r["doc_id"] for r in rows})
    hit_rows = sorted(
        ({"term": t, "class": GAZETTEER[t][0], "owner": GAZETTEER[t][1],
          "docs": len(v["docs"]), "chunks": v["chunks"]} for t, v in hits.items()),
        key=lambda d: (-d["docs"], -d["chunks"]))
    unk_rows = sorted(
        ({"term": t, "docs": len(v["docs"]), "chunks": v["chunks"]}
         for t, v in unknown.items() if len(v["docs"]) >= min_docs),
        key=lambda d: (-d["docs"], -d["chunks"]))[:top]

    _write_tsv(OUT_DIR / "gazetteer_hits.tsv", hit_rows,
               ["term", "class", "owner", "docs", "chunks"])
    _write_tsv(OUT_DIR / "candidate_terms.tsv", unk_rows, ["term", "docs", "chunks"])

    print("\n=== TERM MINING (raw text, gazetteer route) ===")
    print(f"documents            {n_docs:>7,}")
    print(f"chunks               {len(rows):>7,}")
    print(f"chunks with >=1 hit  {chunks_with_hits:>7,} "
          f"({chunks_with_hits / max(1, len(rows)):.0%})")
    print(f"chunks with >=2 hits {multi_hit_chunks:>7,} "
          f"({multi_hit_chunks / max(1, len(rows)):.0%})  <-- LLM relation-pass workload")
    print(f"seed terms defined   {len(GAZETTEER):>7,}")
    print(f"seed terms observed  {len(hit_rows):>7,}")
    print(f"unrecognised >= {min_docs} docs {len(unk_rows):>5,}  <-- gazetteer growth queue")

    by_class: Counter = Counter()
    for h in hit_rows:
        by_class[h["class"]] += 1
    print("\n--- observed vocabulary by class ---")
    for cls, n in by_class.most_common():
        print(f"  {cls:<14} {n:>4}")

    print("\n--- top 45 gazetteer hits ---")
    print(f"{'term':<34}{'class':<13}{'docs':>5}{'chunks':>8}")
    for h in hit_rows[:45]:
        print(f"{h['term'][:33]:<34}{h['class']:<13}{h['docs']:>5}{h['chunks']:>8}")

    print("\n--- top 50 unrecognised candidates (triage into GAZETTEER) ---")
    for u in unk_rows[:50]:
        print(f"  {u['term'][:44]:<46}{u['docs']:>5}{u['chunks']:>8}")

    print(f"\nwrote {OUT_DIR/'gazetteer_hits.tsv'} and {OUT_DIR/'candidate_terms.tsv'}")


# ------------------------------------------------------------------- dupes

_VER_SUFFIX = re.compile(
    r"[\s_\-]*(?:v\d+(?:\.\d+)?|_new_|new|final|copy|draft|rev\d*|"
    r"\d{1,2}\s?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    r"(?:\s?\d{2,4})?)$", re.I)
_WS = re.compile(r"\s+")


def _family_key(file_name: str) -> str:
    """Normalized filename stem: drops extension and version/date suffixes so
    'AI FinOps - Introductory.docx' and 'AI FinOps - Introductory v2.docx'
    collapse to one family. Deterministic, needs no model, and catches the
    .docx/.md pairs that content hashing misses because the renderers differ."""
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", file_name or "")
    stem = _WS.sub(" ", stem.replace("_", " ")).strip()
    prev = None
    while prev != stem:                       # strip stacked suffixes: "... v2 final"
        prev = stem
        stem = _VER_SUFFIX.sub("", stem).strip()
    return stem.lower()


def find_dupes(scope_ids: list[str], jaccard_min: float,
               containment_min: float, max_cluster: int) -> None:
    placeholders = ",".join("?" * len(scope_ids))
    with store.conn() as c:
        docs = c.execute(
            f"SELECT id, file_name, source_uri, quality, chunk_count, "
            f"       source_modified_at, created_at "
            f"FROM documents WHERE conversation_id IN ({placeholders}) "
            f"AND status NOT IN ('failed','skipped')", scope_ids).fetchall()
        chunk_rows = c.execute(
            f"SELECT doc_id, text FROM doc_chunks "
            f"WHERE conversation_id IN ({placeholders})", scope_ids).fetchall()

    meta = {d["id"]: dict(d) for d in docs}
    n_docs = len(meta)
    print(f"\n=== DUPLICATE SCAN v2 ===\ndocuments {n_docs:,}  chunks {len(chunk_rows):,}")

    # Step 1: chunk fingerprints, with boilerplate suppressed ---------------
    # A chunk appearing in a large share of the corpus is a template section.
    # It carries no evidence of copying and, left in, chains unrelated
    # documents into one giant cluster (the v1 failure).
    raw: dict[str, set[int]] = defaultdict(set)
    doc_of_hash: dict[int, set[str]] = defaultdict(set)
    for r in chunk_rows:
        h = hash(_WS.sub(" ", (r["text"] or "").strip().lower()))
        raw[r["doc_id"]].add(h)
        doc_of_hash[h].add(r["doc_id"])

    boiler_cut = max(3, int(0.03 * n_docs))
    boilerplate = {h for h, ds in doc_of_hash.items() if len(ds) >= boiler_cut}
    sets = {d: (hs - boilerplate) for d, hs in raw.items()}
    sets = {d: s for d, s in sets.items() if s}
    print(f"boilerplate chunk-hashes suppressed: {len(boilerplate):,} "
          f"(present in >= {boiler_cut} docs)")

    # Step 2: candidate pairs, only within a shared fingerprint -------------
    inverted: dict[int, list[str]] = defaultdict(list)
    for d, s in sets.items():
        for h in s:
            inverted[h].append(d)
    shared: Counter = Counter()
    for ds in inverted.values():
        if len(ds) < 2 or len(ds) > 12:       # a hash in many docs is still weak
            continue
        ds = sorted(ds)
        for i, a in enumerate(ds):
            for b in ds[i + 1:]:
                shared[(a, b)] += 1

    near: dict[tuple[str, str], float] = {}
    contains: dict[tuple[str, str], float] = {}   # (bigger, smaller) -> ratio
    for (a, b), n in shared.items():
        sa, sb = sets[a], sets[b]
        j = n / len(sa | sb)                      # symmetric — no small-doc bias
        if j >= jaccard_min:
            near[(a, b)] = j
        big, small = (a, b) if len(sa) >= len(sb) else (b, a)
        cov = n / len(sets[small])
        if cov >= containment_min and len(sets[big]) >= 1.5 * len(sets[small]):
            contains[(big, small)] = cov

    # Step 3: filename families --------------------------------------------
    fams: dict[str, list[str]] = defaultdict(list)
    for d, m in meta.items():
        fams[_family_key(m["file_name"])].append(d)
    fam_pairs = {tuple(sorted((a, b)))
                 for members in fams.values() if len(members) > 1
                 for i, a in enumerate(sorted(members))
                 for b in sorted(members)[i + 1:]}

    # Step 4: complete-linkage clustering, size-capped ----------------------
    # A document joins a cluster only if it links to EVERY current member.
    # This is what stops A~B, B~C from silently merging A and C.
    linked: set[tuple[str, str]] = set(near) | fam_pairs
    adj: dict[str, set[str]] = defaultdict(set)
    for a, b in linked:
        adj[a].add(b)
        adj[b].add(a)

    clusters: list[list[str]] = []
    placed: set[str] = set()
    for d in sorted(adj, key=lambda x: -len(adj[x])):
        if d in placed:
            continue
        members = [d]
        for cand in sorted(adj[d], key=lambda x: -len(adj[x])):
            if cand in placed or len(members) >= max_cluster:
                continue
            if all(cand in adj[m] for m in members):
                members.append(cand)
        if len(members) > 1:
            clusters.append(members)
            placed.update(members)

    def survivor(members: list[str]) -> str:
        """quality is uniformly 1.0 in this corpus, so it cannot break ties.
        Rank by recency, but only among documents at least 70% as thorough as
        the biggest in the cluster — a thin new note must not retire a
        comprehensive older one."""
        biggest = max(len(sets.get(m, ())) for m in members)
        eligible = [m for m in members
                    if len(sets.get(m, ())) >= 0.7 * biggest] or members
        return max(eligible, key=lambda m: (
            meta[m].get("source_modified_at") or meta[m].get("created_at") or "",
            meta[m].get("chunk_count") or 0))

    out: list[dict] = []
    retired = 0
    for i, members in enumerate(clusters):
        keep = survivor(members)
        for m in members:
            out.append({
                "cluster": f"C{i:03d}", "role": "KEEP" if m == keep else "SUPERSEDED",
                "reason": "near_duplicate", "file_name": meta[m]["file_name"],
                "modified": (meta[m].get("source_modified_at")
                             or meta[m].get("created_at") or "")[:19],
                "chunks": meta[m].get("chunk_count") or 0, "doc_id": m,
                "source_uri": meta[m].get("source_uri") or ""})
            retired += m != keep

    already = {r["doc_id"] for r in out if r["role"] == "SUPERSEDED"}
    for (big, small), cov in sorted(contains.items(), key=lambda kv: -kv[1]):
        if small in already:
            continue
        already.add(small)
        out.append({
            "cluster": "CONTAIN", "role": "ABSORBED", "reason": f"contained_{cov:.2f}",
            "file_name": meta[small]["file_name"],
            "modified": (meta[small].get("source_modified_at") or "")[:19],
            "chunks": meta[small].get("chunk_count") or 0, "doc_id": small,
            "source_uri": f"absorbed by: {meta[big]['file_name']}"})

    _write_tsv(OUT_DIR / "duplicate_clusters.tsv", out,
               ["cluster", "role", "reason", "file_name", "modified", "chunks",
                "doc_id", "source_uri"])

    sizes = Counter(len(c) for c in clusters)
    print(f"\nnear-duplicate pairs      {len(near):>6,}")
    print(f"containment pairs         {len(contains):>6,}")
    print(f"filename-family pairs     {len(fam_pairs):>6,}")
    print(f"clusters                  {len(clusters):>6,}  sizes {dict(sorted(sizes.items()))}")
    print(f"largest cluster           {max((len(c) for c in clusters), default=0):>6}"
          f"   (v1 produced 140 — anything >8 means the rule is still too loose)")
    print(f"retired as near-duplicate {retired:>6,}")
    print(f"absorbed by containment   {len(contains):>6,}")

    for i, members in enumerate(clusters[:10]):
        keep = survivor(members)
        print(f"\n  C{i:03d}")
        for m in members:
            tag = "KEEP      " if m == keep else "SUPERSEDED"
            print(f"    {tag} {meta[m]['file_name'][:56]:<58}"
                  f"chunks={meta[m].get('chunk_count') or 0:<5}"
                  f"mod={(meta[m].get('source_modified_at') or '?')[:10]}")

    print("\nNOTHING WAS MODIFIED. This is a report, not a migration.")


def main() -> None:
    p = argparse.ArgumentParser(description="Read-only corpus analysis, v2.")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("terms", help="gazetteer match + unknown-term triage queue")
    t.add_argument("--collection", default=None)
    t.add_argument("--min-docs", type=int, default=3)
    t.add_argument("--top", type=int, default=400)

    d = sub.add_parser("dupes", help="near-duplicate + containment detection")
    d.add_argument("--collection", default=None)
    d.add_argument("--jaccard", type=float, default=0.55)
    d.add_argument("--containment", type=float, default=0.85)
    d.add_argument("--max-cluster", type=int, default=8)

    a = p.parse_args()
    scopes = _scopes(a.collection)
    print("scanning: " + ", ".join(t for _, t in scopes))
    ids = [i for i, _ in scopes]

    if a.cmd == "terms":
        mine_terms(ids, a.min_docs, a.top)
    else:
        find_dupes(ids, a.jaccard, a.containment, a.max_cluster)


if __name__ == "__main__":
    main()
