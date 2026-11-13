"""ChatCatchUp - Summarize WhatsApp group chats with topic shifts, arguments, and decisions.
Supports Cloud (Groq) and 100% Local & Offline (Ollama), with PII anonymization.
Run:  streamlit run app.py
"""
import os
from datetime import datetime, time

import streamlit as st

from anonymizer import Anonymizer
from parser import parse_chat, last_n, in_range
from summarizer import summarize, get_available_models, get_ollama_status

st.set_page_config(page_title="WhatsApp ChatCatchUp", page_icon="💬", layout="wide")

st.title("💬 WhatsApp ChatCatchUp")
st.caption("Upload an exported WhatsApp group chat to track topic shifts, arguments made, decisions, and who said what.")

with st.sidebar:
    st.header("⚙️ Engine & Privacy")

    provider_choice = st.radio(
        "Inference Engine",
        ["☁️ Cloud (Groq)", "🔒 Local & Offline (Ollama)"],
        index=0,
        help="Local mode runs 100% on your machine with zero data leaving your computer."
    )

    is_ollama = "Ollama" in provider_choice

    if not is_ollama:
        # Groq Settings
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Get a free API key at https://console.groq.com/keys"
        )
        st.markdown("Free key: [console.groq.com/keys](https://console.groq.com/keys)")

        available_models = get_available_models(api_key)
        default_idx = (
            available_models.index("llama-3.3-70b-versatile")
            if "llama-3.3-70b-versatile" in available_models
            else 0
        )
        model_choice = st.selectbox(
            "Model",
            available_models,
            index=default_idx,
            help="llama-3.3-70b-versatile is the recommended default production model on Groq."
        )
        ollama_host = "http://localhost:11434"
    else:
        # Local Ollama Settings
        api_key = ""
        ollama_host = st.text_input(
            "Ollama Host",
            value=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            help="Default local Ollama URL is http://localhost:11434"
        )

        ollama_online, installed_models = get_ollama_status(ollama_host)
        if ollama_online:
            st.success("🟢 Local Ollama server connected")
            if installed_models:
                model_choice = st.selectbox("Installed Local Model", installed_models)
            else:
                st.warning("No models found locally. Run `ollama run llama3.2:3b` in your terminal.")
                model_choice = st.text_input("Model name", value="llama3.2:3b")
        else:
            st.error("⚠️ Ollama server not detected.")
            st.caption("Start it in your terminal: `ollama serve` or run `ollama run llama3.2:3b`")
            model_choice = st.text_input("Model name", value="llama3.2:3b")

    st.markdown("---")
    st.subheader("🛡️ Privacy & PII Scrubbing")
    enable_anon = st.toggle(
        "Anonymize Personal Information",
        value=True,
        help="Replaces real names, phone numbers, emails, and links with pseudonyms before sending to the model."
    )

    if enable_anon:
        with st.expander("Redaction Options", expanded=False):
            mask_names = st.checkbox("Mask Participant Names", value=True)
            mask_phones = st.checkbox("Mask Phone Numbers", value=True)
            mask_links = st.checkbox("Mask URLs & Google Form links", value=True)
            mask_emails = st.checkbox("Mask Email Addresses", value=True)
            restore_locally = st.checkbox(
                "Restore real names locally on screen",
                value=True,
                help="The model only ever sees 'Participant_1', 'Participant_2'. Real names are swapped back in memory on your computer for display."
            )
    else:
        mask_names = mask_phones = mask_links = mask_emails = False
        restore_locally = False

    st.markdown("---")
    st.markdown("**How to export your chat:**")
    st.markdown("1. Open WhatsApp group chat\n2. Tap **⋮** (or group name on iOS)\n3. Tap **Export Chat**\n4. Select **Without Media**")

file = st.file_uploader("Upload exported chat (.txt)", type=["txt"])

if file:
    raw = file.getvalue().decode("utf-8", errors="ignore")
    messages = parse_chat(raw)

    if not messages:
        st.error("No valid WhatsApp messages found. Please verify this is a valid WhatsApp chat export (.txt) file.")
        st.stop()

    senders = sorted({m.sender for m in messages})
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Messages", len(messages))
    c2.metric("Participants", len(senders))
    c3.metric("Timespan (Days)", (messages[-1].dt.date() - messages[0].dt.date()).days + 1)

    if enable_anon:
        st.info("🛡️ **PII Anonymization Enabled:** Names, phone numbers, emails, and links will be scrubbed before passing to the AI model.")
    elif is_ollama:
        st.success("🔒 **100% Offline Mode:** Running locally via Ollama. No messages will leave this computer.")

    mode = st.radio(
        "What do you want to analyze?",
        ["Last N messages", "Date range", "Whole chat"],
        horizontal=True
    )

    selected = messages
    if mode == "Last N messages":
        if len(messages) <= 20:
            st.caption("Short chat: analyzing all messages.")
        else:
            default_n = min(300, len(messages))
            n = st.slider("Select message count", 20, len(messages), default_n)
            selected = last_n(messages, n)
    elif mode == "Date range":
        first_date, last_date = messages[0].dt.date(), messages[-1].dt.date()
        d1, d2 = st.columns(2)
        start = d1.date_input("From", first_date, min_value=first_date, max_value=last_date)
        end = d2.date_input("To", last_date, min_value=first_date, max_value=last_date)
        selected = in_range(
            messages,
            datetime.combine(start, time.min),
            datetime.combine(end, time.max)
        )

    st.write(f"Analyzing **{len(selected)}** messages.")
    if len(selected) > 1500 and not is_ollama:
        st.warning(
            "Large chat detected (>1500 messages)! If you hit Groq's free-tier rate limits, "
            "try analyzing in smaller date ranges or message batches."
        )

    if st.button("🚀 Analyze & Summarize", type="primary", disabled=not selected):
        if not is_ollama and not api_key:
            st.error("Please provide your Groq API key in the sidebar, or switch to 'Local & Offline (Ollama)'.")
            st.stop()

        status = st.empty()

        # Build Anonymizer if requested
        anon = (
            Anonymizer(
                mask_names=mask_names,
                mask_phones=mask_phones,
                mask_links=mask_links,
                mask_emails=mask_emails,
            )
            if enable_anon
            else None
        )

        try:
            result = summarize(
                selected,
                provider="ollama" if is_ollama else "groq",
                api_key=api_key,
                model=model_choice,
                ollama_host=ollama_host,
                anonymizer=anon,
                restore_names_locally=restore_locally,
                progress=status.info,
            )
        except Exception as e:
            status.empty()
            st.error(f"Error during summarization: {e}")
            st.stop()
        status.empty()

        # 1. Executive Summary
        st.subheader("📋 Executive Summary")
        st.info(result.get("overall_summary", "No summary generated."))

        # 2. Topic Shifts & Arguments Breakdown
        topics = result.get("topics", [])
        if topics:
            st.subheader("💬 Topic Shifts & Debates")
            for idx, t in enumerate(topics, 1):
                title = t.get("topic_title", f"Topic {idx}")
                time_range = t.get("time_range", "")
                header_text = f"**{idx}. {title}** ({time_range})" if time_range else f"**{idx}. {title}**"

                with st.expander(header_text, expanded=True):
                    if t.get("summary"):
                        st.markdown(f"**Topic Overview:** {t['summary']}")

                    # Arguments & Debates
                    debates = t.get("arguments_and_debates", [])
                    if debates:
                        st.markdown("#### ⚖️ Arguments & Debates")
                        for d in debates:
                            st.markdown(f"**Issue / Proposal:** {d.get('issue', 'General Discussion')}")
                            for vp in d.get("viewpoints", []):
                                st.markdown(f"- **{vp.get('person', '?')}**: {vp.get('argument', '')}")
                            outcome = d.get("outcome")
                            if outcome:
                                st.markdown(f"*Outcome:* `{outcome}`")
                            st.markdown("---")

                    # Key contributions
                    contribs = t.get("contributions", [])
                    if contribs:
                        st.markdown("#### 👤 Contributions on this Topic")
                        for c in contribs:
                            st.markdown(f"- **{c.get('name', '?')}**: {c.get('said', '')}")

        # 3. Decisions, Action Items & Open Questions
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("✅ Decisions Made")
            decisions = result.get("decisions", [])
            if decisions:
                for dec in decisions:
                    st.markdown(f"- {dec}")
            else:
                st.markdown("*No explicit decisions recorded.*")

            st.subheader("❓ Open Questions / Unresolved Issues")
            questions = result.get("open_questions", [])
            if questions:
                for q in questions:
                    st.markdown(f"- {q}")
            else:
                st.markdown("*None*")

        with col_right:
            st.subheader("📌 Action Items")
            items = result.get("action_items", [])
            if items:
                for item in items:
                    due_info = f" *(due: {item['due']})*" if item.get("due") else ""
                    st.markdown(f"- **{item.get('owner', 'Unassigned')}**: {item.get('task', '')}{due_info}")
            else:
                st.markdown("*No action items identified.*")
