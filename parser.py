"""Parse WhatsApp exported .txt chats (Android, iOS, 12h, 24h, ISO, European formats)."""
import re
from dataclasses import dataclass
from datetime import datetime

# Matches both Android and iOS formats across different locales:
# Examples:
#   12/10/26, 9:01 pm - Rahul: message
#   12/10/2026, 21:01 - Rahul: message
#   12.10.2026, 21:01 - Rahul: message
#   [12/10/26, 9:01:15 PM] Rahul: message
#   [12/10/2026, 21:01:15] Rahul: message
#   12/10/2026 at 9:01:15 PM - Rahul: message
#   2026-10-12, 21:01 - Rahul: message
#   [2026-10-12, 21:01:15] Rahul: message
LINE_RE = re.compile(
    r"^(?:\[|\b)(\d{1,4})[/.\-](\d{1,2})[/.\-](\d{1,4})(?:,\s*|\s+at\s+|\s+)"
    r"(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([AaPp]\.?[Mm]\.?)?\]?\s*(?:-\s*)?(.*)$"
)

IGNORED_PATTERNS = [
    re.compile(r"^<attached:.*>$", re.I),
    re.compile(r"^<?this message was edited>?$", re.I),
    re.compile(r".*omitted>?$", re.I),
    re.compile(r"^(this message was deleted|you deleted this message|null)$", re.I),
    re.compile(r"^missed (?:group )?(?:voice|video) call\.?$", re.I),
]


@dataclass
class Message:
    dt: datetime
    sender: str
    text: str

    def format(self) -> str:
        return f"[{self.dt:%d %b %H:%M}] {self.sender}: {self.text}"


def _clean(line: str) -> str:
    # Strip invisible Unicode control characters and zero-width spaces
    for ch in ("\u200e", "\u200f", "\ufeff", "\u200b", "\u202a", "\u202c"):
        line = line.replace(ch, "")
    # Normalize non-breaking and narrow spaces
    return line.replace("\u202f", " ").replace("\u00a0", " ").rstrip("\r\n")


def is_ignored(text: str) -> bool:
    clean = text.strip()
    return any(p.search(clean) for p in IGNORED_PATTERNS)


def parse_chat(raw: str) -> list[Message]:
    entries = []  # [d1, d2, d3, hh, mm, ss, ampm, rest]
    for raw_line in raw.splitlines():
        line = _clean(raw_line)
        if not line:
            continue
        m = LINE_RE.match(line)
        if m:
            entries.append(list(m.groups()))
        elif entries:  # continuation of a multi-line message
            entries[-1][7] += "\n" + line

    if not entries:
        return []

    # Detect date format (DD/MM vs MM/DD) for entries where year is at the end
    dayfirst = True
    non_iso = [e for e in entries if len(e[0]) <= 2]
    if any(int(e[0]) > 12 for e in non_iso if e[0].isdigit()):
        dayfirst = True
    elif any(int(e[1]) > 12 for e in non_iso if e[1].isdigit()):
        dayfirst = False

    messages = []
    for d1, d2, d3, hh, mm, ss, ampm, rest in entries:
        try:
            if len(d1) == 4:
                # ISO format: YYYY-MM-DD
                year = int(d1)
                month = int(d2)
                day = int(d3)
            else:
                # DD/MM/YY or MM/DD/YY
                year = int(d3)
                if year < 100:
                    year += 2000
                day, month = (int(d1), int(d2)) if dayfirst else (int(d2), int(d1))

            hour = int(hh)
            if ampm:
                p = ampm.lower().replace(".", "")
                if p == "pm" and hour != 12:
                    hour += 12
                elif p == "am" and hour == 12:
                    hour = 0

            dt = datetime(year, month, day, hour, int(mm), int(ss or 0))
        except (ValueError, TypeError):
            continue

        if ": " not in rest:
            continue  # system message (joined, left, encryption notice...)

        sender, text = rest.split(": ", 1)
        # Clean sender: strip iOS tilde and trailing/leading whitespace
        sender = sender.strip().lstrip("~").strip()
        if not sender or len(sender) > 60:
            continue

        text = text.strip()
        # Strip leading [Forwarded] or media omitted tags attached to real text
        text = re.sub(r"^(?:\[Forwarded\]\s*|<[^>]+omitted>\s*)+", "", text, flags=re.I).strip()
        if not text or is_ignored(text):
            continue

        messages.append(Message(dt, sender, text))

    return messages


def last_n(messages: list[Message], n: int) -> list[Message]:
    return messages[-n:]


def in_range(messages: list[Message], start: datetime, end: datetime) -> list[Message]:
    return [m for m in messages if start <= m.dt <= end]
