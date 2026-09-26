"""`docgraph bench`: does the graph actually help an agent?

A question set (bench_questions.json: where-defined, callers, impact, flow,
architecture, tests-to-run, with reference answers for this repo) is run
through pluggable runners:

  docgraph   offline, no LLM: answers from DocGraph REST (search + context)
  grep       offline, no LLM: answers from a keyword grep of the repo
  search     offline, in-process: retrieval quality only (recall@5 / MRR of
             the reference symbols) -- used to compare embedding models
  telecode   an actual agent: submits CLAUDE_CODE tasks to the telecode Task
             API (engine claude_code, model haiku, is_local false) in two
             modes, `mcp` (agent told to use DocGraph's REST tools via curl)
             and `grep` (agent told to use grep / file reads only)

Scores: keyword recall against the reference keywords (offline), optional
LLM judge (telecode runner, `judge=True`: a second haiku task grades the
answer 0-10 against the reference), plus tokens and tool calls per answer.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

QUESTIONS_FILE = Path(__file__).with_name("bench_questions.json")


def load_questions(path: Path | None = None) -> list[dict]:
    data = json.loads((path or QUESTIONS_FILE).read_text(encoding="utf-8"))
    return data["questions"] if isinstance(data, dict) else data


def keyword_recall(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    low = (answer or "").lower()
    return sum(1 for k in keywords if k.lower() in low) / len(keywords)


def est_tokens(text: str) -> int:
    return max(0, (len(text or "") + 3) // 4)


@dataclass
class Answer:
    text: str
    tokens: int = 0
    tool_calls: int = 0
    cost_usd: float | None = None
    elapsed: float = 0.0
    error: str | None = None
    extra: dict = field(default_factory=dict)


def _get(url: str, timeout: float = 60.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(url: str, body: dict, timeout: float = 60.0):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ---- offline runners ------------------------------------------------------------

class DocGraphRunner:
    """search(question) -> context() of the top hits, via the host REST API."""
    name = "docgraph"

    def __init__(self, host_url: str, root: str | None = None, tokens: int = 1200, top: int = 3):
        self.host = host_url.rstrip("/")
        self.root = root
        self.tokens = tokens
        self.top = top

    def _u(self, path: str, **params) -> str:
        if self.root:
            params["root"] = self.root
        return f"{self.host}{path}?{urllib.parse.urlencode(params)}"

    def answer(self, q: dict) -> Answer:
        t0 = time.time()
        calls = 1
        hits = _get(self._u("/api/search", q=q["question"], limit=8))
        parts: list[str] = []
        seen: set[str] = set()
        for h in hits[: self.top]:
            if h["name"] in seen:
                continue
            seen.add(h["name"])
            ctx = _get(self._u("/api/context", symbol=h["qname"], tokens=self.tokens // self.top))
            calls += 1
            parts.append(ctx.get("text") or "")
        text = "\n".join(parts)
        return Answer(text=text, tokens=est_tokens(text), tool_calls=calls, elapsed=time.time() - t0)


_STOP = {"the", "and", "which", "what", "where", "when", "does", "that", "this", "with", "from",
         "into", "how", "are", "its", "for", "run", "after", "should", "would", "call", "calls",
         "changing", "main", "path", "between", "implemented", "defined", "function", "functions",
         "method", "methods", "tests", "test", "cover", "happens", "client", "tool", "file", "files"}


class GrepRunner:
    """Baseline: grep the repo for the question's identifiers / keywords."""
    name = "grep"

    def __init__(self, repo: Path, max_lines: int = 60, exts=(".py", ".js", ".ts", ".md", ".html")):
        self.repo = Path(repo)
        self.max_lines = max_lines
        self.exts = exts
        self._files = [p for p in self.repo.rglob("*")
                       if p.is_file() and p.suffix in exts and ".venv" not in p.parts
                       and ".docgraph" not in p.parts and ".git" not in p.parts]

    def answer(self, q: dict) -> Answer:
        t0 = time.time()
        words = [w for w in re.findall(r"[A-Za-z_][\w./]*", q["question"])
                 if len(w) > 3 and w.lower() not in _STOP]
        out: list[str] = []
        calls = 0
        for w in words[:6]:
            calls += 1
            pat = re.compile(re.escape(w.split(".")[-1]), re.IGNORECASE)
            n = 0
            for p in self._files:
                if n >= self.max_lines:
                    break
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for i, ln in enumerate(text.splitlines(), 1):
                    if pat.search(ln):
                        out.append(f"{p.relative_to(self.repo).as_posix()}:{i}: {ln.strip()[:160]}")
                        n += 1
                        if n >= self.max_lines:
                            break
        text = "\n".join(out[: self.max_lines * 3])
        return Answer(text=text, tokens=est_tokens(text), tool_calls=calls, elapsed=time.time() - t0)


class SearchRunner:
    """Retrieval-only, in-process: does `search` surface the reference symbols?"""
    name = "search"

    def __init__(self, repo: Path, embedding_model: str | None = None, gpu: bool = False, k: int = 5):
        from docgraph.config import load_config
        from docgraph.db import GraphDB
        from docgraph.embed import Embedder, resolve_device
        from docgraph.retrieve import Retriever
        st = {}
        try:
            st = json.loads((Path(repo) / ".docgraph" / "state.json").read_text())
        except Exception:
            pass
        model = embedding_model or st.get("embedding_model") or "BAAI/bge-small-en-v1.5"
        self.cfg = load_config(Path(repo), embedding_model=model)
        self.db = GraphDB(self.cfg.db_path, read_only=True)
        self.retriever = Retriever(self.db, Embedder(model, device=resolve_device(gpu)), cfg=self.cfg)
        self.k = k
        self.model = model

    def answer(self, q: dict) -> Answer:
        t0 = time.time()
        hits = self.retriever.search(q["question"], limit=20)
        names = [h["name"] for h in hits]
        want = [s for s in q.get("symbols") or []]
        rank = next((i + 1 for i, n in enumerate(names) if n in want), None)
        top = names[: self.k]
        return Answer(text="\n".join(f"{h['name']} {h['file']}" for h in hits[: self.k]),
                      tokens=0, tool_calls=1, elapsed=time.time() - t0,
                      extra={"rank": rank, "recall_at_k": (sum(1 for s in want if s in top) / len(want)) if want else 0.0,
                             "mrr": (1.0 / rank) if rank else 0.0})

    def close(self) -> None:
        try:
            self.db.close()
        except Exception:
            pass


# ---- telecode runner (real agent) -----------------------------------------------

MCP_PROMPT = """You are answering a question about the code repository at {repo}.
Use ONLY the DocGraph code-graph tools, exposed over HTTP (use curl; do not read
or grep source files). Endpoints (GET, JSON, add &root={root} if needed):
  {host}/api/search?q=...            hybrid symbol search
  {host}/api/context?symbol=...      360 view of a symbol (callers, callees, tests, flows)
  {host}/api/call_graph?name=...&depth=2
  {host}/api/impact_of?target=...    blast radius (file path or symbol)
  {host}/api/trace?a=...&b=...       call path between two symbols
  {host}/api/routes                  HTTP routes + MCP tools and their handlers
  {host}/api/clusters                architecture clusters
  {host}/api/detect_changes          impact of the working-tree diff
Answer concisely (at most 120 words), naming files and functions.

Question: {question}"""

GREP_PROMPT = """You are answering a question about the code repository at {repo}.
Use only grep / file search and reading files inside that directory (no web,
no other tools). Answer concisely (at most 120 words), naming files and functions.

Question: {question}"""

JUDGE_PROMPT = """Grade an answer to a question about a codebase against the reference.
Question: {question}
Reference answer: {reference}
Candidate answer: {answer}
Reply with ONLY a JSON object: {{"score": <integer 0-10>, "reason": "<one sentence>"}}"""


class TelecodeRunner:
    """Submits CLAUDE_CODE tasks to telecode (POST /api/tasks). Cloud only:
    engine claude_code, model haiku by default, is_local false -- never the
    local llama path."""
    name = "telecode"

    def __init__(self, repo: Path, mode: str = "mcp", host_url: str = "http://127.0.0.1:5500",
                 root: str | None = None, telecode_url: str = "http://127.0.0.1:1235",
                 model: str = "haiku", timeout: float = 600.0, judge: bool = False):
        if mode not in ("mcp", "grep"):
            raise ValueError("mode must be mcp or grep")
        self.repo = Path(repo).resolve()
        self.mode = mode
        self.name = f"telecode-{mode}"
        self.host = host_url.rstrip("/")
        self.root = root or ""
        self.tc = telecode_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.judge = judge

    def _run_task(self, prompt: str) -> dict:
        body = {"task_type": "CLAUDE_CODE",
                "params": {"prompt": prompt, "is_local": False, "model": self.model},
                "metadata": {"source": "docgraph-bench"},
                "task_timeout_seconds": int(self.timeout)}
        sub = _post(f"{self.tc}/api/tasks", body, timeout=30)
        if not sub.get("success"):
            raise RuntimeError(sub.get("error") or "submit failed")
        tid = sub["task_id"]
        deadline = time.time() + self.timeout + 60
        while time.time() < deadline:
            rec = _get(f"{self.tc}/api/tasks/{tid}", timeout=30)
            if rec.get("status") in ("completed", "failed", "cancelled"):
                return rec
            time.sleep(3)
        raise TimeoutError(f"task {tid} did not finish")

    @staticmethod
    def _usage(rec: dict) -> tuple[int, int, float | None]:
        tokens = tools = 0
        cost = None
        for ev in (rec.get("metadata") or {}).get("events") or []:
            if ev.get("kind") == "done":
                tokens = int(ev.get("input_tokens") or 0) + int(ev.get("output_tokens") or 0)
                tools = int(ev.get("tool_count") or 0)
                cost = ev.get("cost_usd")
        return tokens, tools, cost

    def answer(self, q: dict) -> Answer:
        t0 = time.time()
        tmpl = MCP_PROMPT if self.mode == "mcp" else GREP_PROMPT
        prompt = tmpl.format(repo=str(self.repo), host=self.host, root=self.root,
                             question=q["question"])
        try:
            rec = self._run_task(prompt)
        except Exception as exc:  # noqa: BLE001
            return Answer(text="", error=str(exc), elapsed=time.time() - t0)
        res = rec.get("result") or {}
        text = res.get("result") if isinstance(res, dict) else str(res)
        tokens, tools, cost = self._usage(rec)
        ans = Answer(text=text or "", tokens=tokens, tool_calls=tools, cost_usd=cost,
                     elapsed=time.time() - t0,
                     error=None if rec.get("status") == "completed" else rec.get("status"))
        if self.judge and ans.text:
            try:
                jr = self._run_task(JUDGE_PROMPT.format(question=q["question"],
                                                        reference=q["reference"], answer=ans.text))
                jtext = (jr.get("result") or {}).get("result") or ""
                m = re.search(r"\{.*\}", jtext, re.DOTALL)
                if m:
                    j = json.loads(m.group(0))
                    ans.extra["judge_score"] = float(j.get("score", 0)) / 10.0
                    ans.extra["judge_reason"] = j.get("reason", "")
            except Exception as exc:  # noqa: BLE001
                ans.extra["judge_error"] = str(exc)
        return ans


# ---- driver ----------------------------------------------------------------------

def run(questions: list[dict], runners: list, progress=None) -> dict:
    rows: list[dict] = []
    for q in questions:
        for r in runners:
            a = r.answer(q)
            row = {"id": q["id"], "category": q["category"], "runner": r.name,
                   "recall": round(keyword_recall(a.text, q.get("keywords") or []), 3),
                   "tokens": a.tokens, "tool_calls": a.tool_calls, "cost_usd": a.cost_usd,
                   "elapsed": round(a.elapsed, 2), "error": a.error, **a.extra,
                   "answer": (a.text or "")[:1500]}
            rows.append(row)
            if progress:
                progress(row)
    summary: dict[str, dict] = {}
    for r in runners:
        mine = [x for x in rows if x["runner"] == r.name]
        if not mine:
            continue
        n = len(mine)
        s = {"n": n,
             "recall": round(sum(x["recall"] for x in mine) / n, 3),
             "tokens": round(sum(x["tokens"] for x in mine) / n, 1),
             "tool_calls": round(sum(x["tool_calls"] for x in mine) / n, 2),
             "errors": sum(1 for x in mine if x["error"])}
        if any("judge_score" in x for x in mine):
            js = [x["judge_score"] for x in mine if "judge_score" in x]
            s["judge"] = round(sum(js) / len(js), 3)
        if any("mrr" in x for x in mine):
            s["mrr"] = round(sum(x.get("mrr", 0.0) for x in mine) / n, 3)
            s["recall_at_k"] = round(sum(x.get("recall_at_k", 0.0) for x in mine) / n, 3)
        costs = [x["cost_usd"] for x in mine if x.get("cost_usd") is not None]
        if costs:
            s["cost_usd"] = round(sum(costs), 4)
        by_cat: dict[str, list[float]] = {}
        for x in mine:
            by_cat.setdefault(x["category"], []).append(x["recall"])
        s["by_category"] = {k: round(sum(v) / len(v), 3) for k, v in by_cat.items()}
        summary[r.name] = s
    return {"summary": summary, "rows": rows}
