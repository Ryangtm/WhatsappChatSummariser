"""Summarize parsed WhatsApp messages via Cloud (Groq) or 100% Local & Offline (Ollama).
Supports topic shifts, debates, contributions, decisions, and privacy anonymization.
"""
import json
import os
import re
import time

try:
    from groq import BadRequestError, APIStatusError, Groq, RateLimitError
except ImportError:
    BadRequestError = APIStatusError = RateLimitError = Exception
    Groq = None

# Default models
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

MAX_CHARS = int(os.getenv("CHUNK_CHARS", "8000"))
MERGE_CHARS = int(os.getenv("MERGE_CHARS", "12000"))
MAX_OUTPUT_TOKENS = 4096

SCHEMA = """Return ONLY valid JSON (no markdown formatting, no code fences, NO trailing commas) matching this structure:
{
  "overall_summary": "3-4 sentence overview of the conversation",
  "topics": [
    {
      "topic_title": "Descriptive title for this topic/discussion",
      "time_range": "e.g. 12 Oct 09:00 - 11:30",
      "summary": "Concise summary of what was discussed",
      "arguments_and_debates": [
        {
          "issue": "What was being debated or proposed",
          "viewpoints": [
            {"person": "Name", "argument": "Their stance, reasoning, or proposal"}
          ],
          "outcome": "Resolution, consensus, deadlock, or postponed"
        }
      ],
      "contributions": [
        {"name": "Participant name", "said": "Summary of what they contributed or argued"}
      ]
    }
  ],
  "decisions": ["Concrete decisions agreed upon"],
  "action_items": [{"owner": "Name or 'Unassigned'", "task": "Action item", "due": "Deadline or null"}],
  "open_questions": ["Unanswered questions, unresolved issues, or pending items"]
}"""

SYSTEM = f"""You are an expert conversation analyst summarizing exported WhatsApp group chats.
Analyze the conversation thoroughly:
1. Detect where topics change: segment the conversation into distinct topics with timeframes.
2. Capture arguments and debates: note who argued what, contrasting viewpoints, and the outcome.
3. Capture participant contributions: summarize what each person said or did under that topic.
4. Extract concrete decisions, action items (with owner and deadline if mentioned), and unanswered questions.
- Preserve exact participant names or aliases. Keep topics and points chronological.
- If chats mix English, Hindi, or Hinglish, write the summary in clear English.
- Be completely faithful to the chat text. Do not invent facts.
- Output STRICT RFC 8259 JSON. Never include trailing commas after the last item in an object or array.
{SCHEMA}"""

MERGE_SYSTEM = f"""You will receive JSON summaries of consecutive parts of a WhatsApp chat.
Merge them into ONE coherent, comprehensive master summary of the whole chat:
- Provide an overarching overall_summary.
- Merge the timeline of topics: merge continuous topics, preserve distinct topic shifts.
- Consolidate arguments_and_debates across topics, preserving key viewpoints and resolutions.
- Consolidate participant contributions under each topic without repeating trivial points.
- Consolidate decisions and action items. Remove any open_questions that were resolved in later parts.
{SCHEMA}"""


def _wait_seconds(err, attempt):
    """Parse Groq's retry-after header / error message or use exponential backoff."""
    m = re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", str(err))
    if m:
        return min(float(m.group(1) or 0) * 60 + float(m.group(2)) + 1, 90)
    return min(5 * (attempt + 1), 45)


def _call_groq(client, system, user, model=DEFAULT_GROQ_MODEL, retries=5):
    json_mode = True
    for attempt in range(retries):
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except RateLimitError as e:
            if attempt == retries - 1:
                raise RuntimeError(
                    f"Groq rate limit reached for {model}. Wait a minute and try again, "
                    "or select fewer messages."
                ) from e
            time.sleep(_wait_seconds(e, attempt))
        except BadRequestError as e:
            if "terms" in str(e).lower():
                raise RuntimeError(
                    f"The model '{model}' requires special terms acceptance on Groq. "
                    "Please select 'llama-3.3-70b-versatile' from the sidebar."
                ) from e
            if json_mode and "json" in str(e).lower():
                json_mode = False
                continue
            raise
        except APIStatusError as e:
            if e.status_code == 404 and "model" in str(e).lower():
                raise RuntimeError(
                    f"Model '{model}' is not available or has been decommissioned on Groq. "
                    "Please select another active model (such as 'llama-3.3-70b-versatile') from the sidebar."
                ) from e
            if e.status_code == 413:
                raise RuntimeError(
                    "Request too large for Groq API. Select fewer messages or reduce chunk size."
                ) from e
            raise
    raise RuntimeError("Groq request failed after retries.")


def _call_ollama(system, user, model=DEFAULT_OLLAMA_MODEL, host=DEFAULT_OLLAMA_HOST, timeout=180):
    """Execute a 100% local, offline inference request to a running Ollama server."""
    import requests

    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code == 404:
            raise RuntimeError(
                f"Model '{model}' not found in local Ollama. "
                f"Please run 'ollama run {model}' in your terminal to download it."
            )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Cannot connect to local Ollama server at {host}. "
            "Make sure Ollama is installed and running ('ollama serve' in your terminal)."
        ) from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama request error: {e}") from e


def _call_llm(provider, client_or_host, system, user, model):
    if provider == "ollama":
        return _call_ollama(system, user, model=model, host=client_or_host)
    else:
        return _call_groq(client_or_host, system, user, model=model)


def _strip_comments_and_trailing_commas(s: str) -> str:
    """Strip JavaScript-style comments and illegal trailing commas outside string literals."""
    # Pass 1: strip // and /* ... */ comments
    pass1 = []
    in_str = False
    esc = False
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if esc:
            pass1.append(c)
            esc = False
            i += 1
            continue
        if c == "\\":
            pass1.append(c)
            if in_str:
                esc = True
            i += 1
            continue
        if c == "\"":
            in_str = not in_str
            pass1.append(c)
            i += 1
            continue
        if not in_str:
            if c == "/" and i + 1 < n and s[i + 1] == "/":
                while i < n and s[i] != "\n":
                    i += 1
                continue
            if c == "/" and i + 1 < n and s[i + 1] == "*":
                i += 2
                while i + 1 < n and not (s[i] == "*" and s[i + 1] == "/"):
                    i += 1
                i += 2
                continue
        pass1.append(c)
        i += 1
    s1 = "".join(pass1)

    # Pass 2: strip trailing commas before } or ]
    pass2 = []
    in_str = False
    esc = False
    i = 0
    n = len(s1)
    while i < n:
        c = s1[i]
        if esc:
            pass2.append(c)
            esc = False
            i += 1
            continue
        if c == "\\":
            pass2.append(c)
            if in_str:
                esc = True
            i += 1
            continue
        if c == "\"":
            in_str = not in_str
            pass2.append(c)
            i += 1
            continue
        if not in_str and c == ",":
            j = i + 1
            while j < n and s1[j] in " \t\r\n":
                j += 1
            if j < n and s1[j] in "]}":
                i += 1
                continue
        pass2.append(c)
        i += 1
    return "".join(pass2)


def _parse_json(text):
    """Safely parse JSON, auto-repairing comments, trailing commas, or truncated brackets."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start == -1:
        raise ValueError("The model did not return a valid JSON object.")
    text = text[start:]

    cleaned = _strip_comments_and_trailing_commas(text)

    # 1. Try standard load
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 2. Try substring up to last closing brace
    end = cleaned.rfind("}")
    if end != -1:
        try:
            return json.loads(_strip_comments_and_trailing_commas(cleaned[:end + 1]))
        except Exception:
            pass

    # 3. Stack-based repair for truncated outputs
    in_string = False
    escape = False
    stack = []
    repaired_chars = []

    for ch in cleaned:
        repaired_chars.append(ch)
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == "\"":
            in_string = not in_string
            continue
        if not in_string:
            if ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if stack and stack[-1] == ch:
                    stack.pop()

    if in_string:
        repaired_chars.append("\"")
    while stack:
        repaired_chars.append(stack.pop())

    repaired = "".join(repaired_chars)
    repaired = _strip_comments_and_trailing_commas(repaired)
    return json.loads(repaired)


def _normalize(d):
    """Ensure the parsed dictionary conforms to the expected UI schema."""
    d = d if isinstance(d, dict) else {}

    def as_list(x):
        return x if isinstance(x, list) else ([x] if x else [])

    topics = []
    for t in as_list(d.get("topics")):
        if not isinstance(t, dict):
            continue
        debates = []
        for deb in as_list(t.get("arguments_and_debates")):
            if isinstance(deb, dict):
                vps = []
                for v in as_list(deb.get("viewpoints")):
                    if isinstance(v, dict):
                        vps.append({
                            "person": str(v.get("person") or "?"),
                            "argument": str(v.get("argument") or "")
                        })
                debates.append({
                    "issue": str(deb.get("issue") or ""),
                    "viewpoints": vps,
                    "outcome": str(deb.get("outcome") or "")
                })

        contribs = []
        for c in as_list(t.get("contributions")):
            if isinstance(c, dict):
                contribs.append({
                    "name": str(c.get("name") or "?"),
                    "said": str(c.get("said") or "")
                })

        topics.append({
            "topic_title": str(t.get("topic_title") or "Discussion Topic"),
            "time_range": str(t.get("time_range") or ""),
            "summary": str(t.get("summary") or ""),
            "arguments_and_debates": debates,
            "contributions": contribs
        })

    actions = []
    for a in as_list(d.get("action_items")):
        if isinstance(a, dict):
            actions.append({
                "owner": str(a.get("owner") or "Unassigned"),
                "task": str(a.get("task") or ""),
                "due": str(a.get("due")) if a.get("due") not in (None, "null", "") else None
            })
        elif a:
            actions.append({"owner": "Unassigned", "task": str(a), "due": None})

    return {
        "overall_summary": str(d.get("overall_summary") or ""),
        "topics": topics,
        "decisions": [str(x) for x in as_list(d.get("decisions")) if x],
        "action_items": actions,
        "open_questions": [str(x) for x in as_list(d.get("open_questions")) if x],
    }


def _chunk(messages, max_chars=MAX_CHARS, pause_hours=2.0):
    """Chunk messages intelligently at natural conversation pauses (>2h) or max character limit."""
    chunks = []
    cur = []
    size = 0
    prev_dt = None

    for m in messages:
        line = m.format()[:max_chars]
        line_len = len(line) + 1

        is_pause = False
        if prev_dt and (m.dt - prev_dt).total_seconds() > pause_hours * 3600:
            is_pause = True

        if cur and ((size + line_len > max_chars) or (is_pause and size >= 4000)):
            chunks.append(cur)
            cur = []
            size = 0

        cur.append(line)
        size += line_len
        prev_dt = m.dt

    if cur:
        chunks.append(cur)
    return chunks


def _summarize_chunk(provider, client_or_host, lines, model):
    out = _call_llm(provider, client_or_host, SYSTEM, "Chat messages:\n\n" + "\n".join(lines), model=model)
    try:
        return _parse_json(out)
    except Exception:
        retry_prompt = (
            "Chat messages:\n\n" + "\n".join(lines)
            + "\n\nReminder: Output ONLY valid RFC 8259 JSON. Do not include trailing commas."
        )
        out = _call_llm(provider, client_or_host, SYSTEM, retry_prompt, model=model)
        try:
            return _parse_json(out)
        except Exception:
            return {
                "overall_summary": out[:500] if out else "Unable to parse complete summary for this section.",
                "topics": [],
                "decisions": [],
                "action_items": [],
                "open_questions": []
            }


def _merge_all(provider, client_or_host, partials, model, progress=None):
    """Merge partial summaries in rounds so the merge prompt never exceeds context limits."""
    while len(partials) > 1:
        groups, cur, size = [], [], 0
        for p in partials:
            s = len(json.dumps(p, ensure_ascii=False))
            if cur and size + s > MERGE_CHARS:
                groups.append(cur)
                cur, size = [], 0
            cur.append(p)
            size += s
        if cur:
            groups.append(cur)

        if len(groups) == len(partials):
            groups = [partials[i:i + 2] for i in range(0, len(partials), 2)]

        merged = []
        for g in groups:
            if len(g) == 1:
                merged.append(g[0])
                continue
            if progress:
                progress(f"Merging summaries ({len(partials)} remaining)...")
            out = _call_llm(provider, client_or_host, MERGE_SYSTEM, json.dumps(g, ensure_ascii=False), model=model)
            try:
                merged.append(_parse_json(out))
            except Exception:
                merged.extend(g)
        partials = merged

    return partials[0]


def get_available_models(api_key=None):
    """Fetch active chat models available to this Groq API key, or return sensible defaults."""
    fallback = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key or Groq is None:
        return fallback
    try:
        client = Groq(api_key=key)
        resp = client.models.list()

        ALLOWED_FAMILIES = ("llama", "qwen", "gemma", "mixtral", "gpt-oss")
        EXCLUDED_KEYWORDS = (
            "orpheus", "canopylabs", "whisper", "guard", "safeguard",
            "embed", "audio", "tts", "stt", "speech", "vision"
        )

        chat_models = []
        for m in getattr(resp, "data", []):
            mid = getattr(m, "id", "")
            lower = mid.lower()
            if any(skip in lower for skip in EXCLUDED_KEYWORDS):
                continue
            if any(fam in lower for fam in ALLOWED_FAMILIES):
                chat_models.append(mid)

        if "llama-3.3-70b-versatile" in chat_models:
            chat_models.remove("llama-3.3-70b-versatile")
            chat_models.insert(0, "llama-3.3-70b-versatile")

        return chat_models if chat_models else fallback
    except Exception:
        return fallback


def get_ollama_status(host=DEFAULT_OLLAMA_HOST):
    """Check if Ollama server is running and return (is_running, models_list)."""
    import requests
    try:
        resp = requests.get(f"{host.rstrip('/')}/api/tags", timeout=2)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", []) if m.get("name")]
            return True, models
    except Exception:
        pass
    return False, []


def summarize(
    messages,
    provider="groq",
    api_key=None,
    model=None,
    ollama_host=DEFAULT_OLLAMA_HOST,
    anonymizer=None,
    restore_names_locally=True,
    progress=None,
):
    """Summarize messages using either Cloud Groq or Local Ollama, with optional PII anonymization."""
    if not messages:
        raise ValueError("No messages to summarize.")

    # 1. PII Anonymization (if enabled)
    processing_messages = messages
    if anonymizer:
        processing_messages = anonymizer.anonymize_messages(messages)

    # 2. Setup Client / Backend
    if provider == "ollama":
        chosen_model = model or DEFAULT_OLLAMA_MODEL
        client_or_host = ollama_host
    else:
        if Groq is None:
            raise ImportError("The 'groq' package is not installed. Please install it using: pip install groq")
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("Missing Groq API key.")
        chosen_model = model or DEFAULT_GROQ_MODEL
        client_or_host = Groq(api_key=key)

    chunks = _chunk(processing_messages)

    # 3. Summarize Chunks
    partials = []
    for i, chunk in enumerate(chunks, 1):
        if progress:
            progress(f"Summarizing part {i} of {len(chunks)} using {chosen_model} ({provider})...")
        partials.append(_summarize_chunk(provider, client_or_host, chunk, model=chosen_model))

    # 4. Merge Chunks
    result = (
        partials[0]
        if len(partials) == 1
        else _merge_all(provider, client_or_host, partials, model=chosen_model, progress=progress)
    )

    normalized = _normalize(result)

    # 5. Local De-anonymization (restore real names only on local display)
    if anonymizer and restore_names_locally:
        normalized = anonymizer.de_anonymize_result(normalized)

    return normalized
