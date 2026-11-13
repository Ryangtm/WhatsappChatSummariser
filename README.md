# ChatCatchUp 💬

Summarize WhatsApp group chats with a focus on **privacy**, **topic shifts**, **debates**, and **action items**.

## Key Features

- **Dual Engine (Cloud or 100% Offline)**:
  - ☁️ **Cloud (Groq)**: Blazing fast inference via `llama-3.3-70b-versatile`.
  - 🔒 **100% Local (Ollama)**: Completely offline and air-gapped on your computer. Zero data leaves your machine.
- **🛡️ Built-in PII Anonymization**:
  - Automatically scrubs phone numbers, emails, URLs/form links, and real participant names.
  - Generates pseudonyms (`Participant_1`, `Participant_2`) for AI inference.
  - Optionally restores real names locally on your screen in memory.
- **Topic Shifts & Debates Extraction**:
  - Automatically segments conversation by topic transitions.
  - Details what issues were debated, who held which viewpoints, and the final resolution.
  - Summarizes per-topic participant contributions.
- **Universal WhatsApp Parser**:
  - Supports Android, iOS, 12h/24h, ISO dates (`YYYY-MM-DD`), and European dot dates.
  - Automatically strips media tags (`<image omitted>`, `<attached: ...>`).

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Choose Your Engine

#### Option A: 100% Local & Offline (Ollama)
No API keys needed!
1. Start Ollama and pull a model:
   ```bash
   ollama run llama3.2:3b
   ```
2. Start ChatCatchUp:
   ```bash
   streamlit run app.py
   ```
3. Select **"🔒 Local & Offline (Ollama)"** in the sidebar.

#### Option B: Cloud (Groq)
1. Get a free API key at [console.groq.com/keys](https://console.groq.com/keys).
2. Start ChatCatchUp:
   ```bash
   export GROQ_API_KEY="your-key"
   streamlit run app.py
   ```
3. Keep **"🛡️ Anonymize Personal Info"** enabled to scrub sensitive data before it reaches the cloud.

---

## How to Export Chat
1. Open the WhatsApp group chat on your phone.
2. Tap **⋮** (Android) or the group name (iOS) -> **More** -> **Export chat**.
3. Select **Without media**.
4. Upload the resulting `.txt` file into ChatCatchUp.

---

## Files
- `anonymizer.py` - PII redactor (masks names, phone numbers, emails, links; restores names locally).
- `parser.py` - Universal WhatsApp export parser (Android, iOS, 12h/24h, ISO, attachment cleanup).
- `summarizer.py` - Topic shift & debate extraction, supporting both Groq and Ollama engines with resilient JSON auto-repair.
- `app.py` - Streamlit UI with privacy controls, engine selector, and interactive breakdown.
