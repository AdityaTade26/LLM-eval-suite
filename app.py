"""
AI Assistant Evaluation App
Compares OSS (Qwen2.5) vs Frontier (Llama-3.3-70B via Groq) side-by-side.
"""

import streamlit as st
from src.oss_assistant import OSSAssistant
from src.frontier_assistant import FrontierAssistant

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Assistant Eval",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0d0d0d;
    --surface: #161616;
    --border: #2a2a2a;
    --accent-oss: #00e5a0;
    --accent-frontier: #7b61ff;
    --text: #f0f0f0;
    --muted: #888;
}

.stApp { background-color: var(--bg); font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.model-badge-oss {
    display: inline-block;
    background: rgba(0,229,160,0.12);
    color: var(--accent-oss);
    border: 1px solid var(--accent-oss);
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 12px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 1px;
}

.model-badge-frontier {
    display: inline-block;
    background: rgba(123,97,255,0.12);
    color: var(--accent-frontier);
    border: 1px solid var(--accent-frontier);
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 12px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 1px;
}

.chat-bubble-user {
    background: #1e1e1e;
    border: 1px solid #2a2a2a;
    border-radius: 12px 12px 2px 12px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 14px;
    color: #f0f0f0;
}

.chat-bubble-assistant-oss {
    background: rgba(0,229,160,0.06);
    border: 1px solid rgba(0,229,160,0.2);
    border-radius: 12px 12px 12px 2px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 14px;
    color: #e0e0e0;
}

.chat-bubble-assistant-frontier {
    background: rgba(123,97,255,0.06);
    border: 1px solid rgba(123,97,255,0.2);
    border-radius: 12px 12px 12px 2px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 14px;
    color: #e0e0e0;
}

.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 18px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
def init_state():
    if "oss_history" not in st.session_state:
        st.session_state.oss_history = []
    if "frontier_history" not in st.session_state:
        st.session_state.frontier_history = []
    if "oss_latencies" not in st.session_state:
        st.session_state.oss_latencies = []
    if "frontier_latencies" not in st.session_state:
        st.session_state.frontier_latencies = []

init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Config")
    st.divider()

    hf_token = st.text_input(
        "HuggingFace API Token",
        type="password",
        help="Required for OSS model (Qwen2.5 via HF Inference API)",
        placeholder="hf_..."
    )
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Free at console.groq.com — required for Frontier model",
        placeholder="gsk_..."
    )

    st.divider()
    st.markdown("**OSS Model**")
    oss_model = st.selectbox(
        "OSS Model",
        ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "microsoft/Phi-3-mini-4k-instruct"],
        label_visibility="collapsed"
    )
    st.markdown('<span class="model-badge-oss">OPEN SOURCE</span>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown("**Frontier Model**")
    frontier_model = st.selectbox(
        "Frontier Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"],
        label_visibility="collapsed"
    )
    st.markdown('<span class="model-badge-frontier">FRONTIER API</span>&nbsp;<span style="font-size:11px;color:#888">Groq — free tier</span>', unsafe_allow_html=True)

    st.divider()
    max_tokens = st.slider("Max tokens", 100, 1000, 512, step=50)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.05)

    st.divider()
    if st.button("🗑️ Clear all history", use_container_width=True):
        st.session_state.oss_history = []
        st.session_state.frontier_history = []
        st.session_state.oss_latencies = []
        st.session_state.frontier_latencies = []
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:1.8rem; margin-bottom:4px;'>AI Assistant Eval</h1>
<p style='color:#888; margin-top:0; font-size:14px;'>
  OSS (Qwen2.5) vs Frontier (Llama-3.3-70B) — hallucination · bias · safety · latency
</p>
""", unsafe_allow_html=True)

# ── Stats bar ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    turns = len(st.session_state.oss_history)
    st.markdown(f'<div class="stat-card"><div style="font-size:22px;font-weight:700;color:#00e5a0">{turns}</div><div style="font-size:11px;color:#888">CONVERSATION TURNS</div></div>', unsafe_allow_html=True)
with col2:
    avg_oss = f"{sum(st.session_state.oss_latencies)/len(st.session_state.oss_latencies):.1f}s" if st.session_state.oss_latencies else "—"
    st.markdown(f'<div class="stat-card"><div style="font-size:22px;font-weight:700;color:#00e5a0">{avg_oss}</div><div style="font-size:11px;color:#888">OSS AVG LATENCY</div></div>', unsafe_allow_html=True)
with col3:
    avg_fr = f"{sum(st.session_state.frontier_latencies)/len(st.session_state.frontier_latencies):.1f}s" if st.session_state.frontier_latencies else "—"
    st.markdown(f'<div class="stat-card"><div style="font-size:22px;font-weight:700;color:#7b61ff">{avg_fr}</div><div style="font-size:11px;color:#888">FRONTIER AVG LATENCY</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="stat-card"><div style="font-size:22px;font-weight:700;color:#f0f0f0">2</div><div style="font-size:11px;color:#888">MODELS ACTIVE</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Side-by-Side Chat", "🧪 Evaluation Runner", "📊 Results"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Side-by-side chat
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_oss, col_frontier = st.columns(2)

    with col_oss:
        st.markdown('<span class="model-badge-oss">OSS</span>&nbsp; <b style="font-size:13px">Qwen2.5</b>', unsafe_allow_html=True)
        chat_oss = st.container(height=420)
        with chat_oss:
            for msg in st.session_state.oss_history:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-bubble-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    latency_tag = f' <span style="font-size:10px;color:#555">{msg.get("latency","")}</span>' if msg.get("latency") else ""
                    st.markdown(f'<div class="chat-bubble-assistant-oss">🟢 {msg["content"]}{latency_tag}</div>', unsafe_allow_html=True)

    with col_frontier:
        st.markdown('<span class="model-badge-frontier">FRONTIER</span>&nbsp; <b style="font-size:13px">Llama-3.3-70B (Groq)</b>', unsafe_allow_html=True)
        chat_fr = st.container(height=420)
        with chat_fr:
            for msg in st.session_state.frontier_history:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-bubble-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    latency_tag = f' <span style="font-size:10px;color:#555">{msg.get("latency","")}</span>' if msg.get("latency") else ""
                    st.markdown(f'<div class="chat-bubble-assistant-frontier">🟣 {msg["content"]}{latency_tag}</div>', unsafe_allow_html=True)

    user_input = st.chat_input("Send a message to both assistants…")

    if user_input:
        if not hf_token or not groq_key:
            st.warning("⚠️ Please enter both API keys in the sidebar to start chatting.")
        else:
            oss = OSSAssistant(model_id=oss_model, hf_token=hf_token, max_tokens=max_tokens, temperature=temperature)
            frontier = FrontierAssistant(model_id=frontier_model, api_key=groq_key, max_tokens=max_tokens, temperature=temperature)

            st.session_state.oss_history.append({"role": "user", "content": user_input})
            st.session_state.frontier_history.append({"role": "user", "content": user_input})

            with st.spinner("Getting responses from both models…"):
                oss_resp = oss.chat(st.session_state.oss_history[:-1], user_input)
                frontier_resp = frontier.chat(st.session_state.frontier_history[:-1], user_input)

            st.session_state.oss_history.append({
                "role": "assistant",
                "content": oss_resp["text"],
                "latency": f'{oss_resp["latency"]:.1f}s'
            })
            st.session_state.frontier_history.append({
                "role": "assistant",
                "content": frontier_resp["text"],
                "latency": f'{frontier_resp["latency"]:.1f}s'
            })
            st.session_state.oss_latencies.append(oss_resp["latency"])
            st.session_state.frontier_latencies.append(frontier_resp["latency"])
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Evaluation Runner
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🧪 Automated Evaluation Suite")
    st.markdown("Run curated prompts across both models and score with LLM-as-judge.")

    from eval.eval_runner import EVAL_PROMPTS

    col_a, col_b = st.columns([2, 1])
    with col_a:
        selected_categories = st.multiselect(
            "Prompt categories",
            ["factual", "adversarial", "bias", "safety"],
            default=["factual", "adversarial", "bias", "safety"]
        )
    with col_b:
        run_eval = st.button("▶ Run Evaluation", type="primary", use_container_width=True)

    filtered = [p for p in EVAL_PROMPTS if p["category"] in selected_categories]
    st.markdown(f"**{len(filtered)} prompts** selected")

    with st.expander("Preview prompts"):
        for p in filtered:
            cat_color = {"factual": "#00e5a0", "adversarial": "#ff6b6b", "bias": "#ffd93d", "safety": "#ff9f43"}.get(p["category"], "#888")
            st.markdown(f'<span style="background:{cat_color}22;color:{cat_color};border-radius:3px;padding:1px 7px;font-size:11px;font-family:monospace">{p["category"].upper()}</span> {p["prompt"]}', unsafe_allow_html=True)

    if run_eval:
        if not hf_token or not groq_key:
            st.warning("⚠️ Please enter both API keys in the sidebar first.")
        else:
            from eval.eval_runner import run_evaluation
            with st.spinner("Running evaluation — this may take a minute…"):
                results = run_evaluation(
                    prompts=filtered,
                    hf_token=hf_token,
                    groq_key=groq_key,
                    oss_model=oss_model,
                    frontier_model=frontier_model,
                    max_tokens=max_tokens
                )
            st.session_state["eval_results"] = results
            st.success(f"✅ Evaluation complete — {len(results)} prompts scored")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Results
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Evaluation Results")

    results = st.session_state.get("eval_results", None)

    if results is None:
        st.info("Run the evaluation in the **Evaluation Runner** tab first.")
    else:
        import pandas as pd
        import json

        df = pd.DataFrame(results)

        st.markdown("#### Score Summary by Category")
        summary = df.groupby("category")[["oss_score", "frontier_score"]].mean().round(2)
        summary.columns = ["OSS (Qwen2.5)", "Frontier (Llama-3.3-70B)"]
        st.dataframe(summary, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("OSS Overall Score", f"{df['oss_score'].mean():.2f}/10")
        with col2:
            st.metric("Frontier Overall Score", f"{df['frontier_score'].mean():.2f}/10")

        st.divider()
        st.markdown("#### Per-Prompt Results")
        for i, row in df.iterrows():
            with st.expander(f'[{row["category"].upper()}] {row["prompt"][:80]}…'):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f'**OSS Response** (score: {row["oss_score"]}/10)')
                    st.markdown(f'<div class="chat-bubble-assistant-oss">{row["oss_response"]}</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'**Frontier Response** (score: {row["frontier_score"]}/10)')
                    st.markdown(f'<div class="chat-bubble-assistant-frontier">{row["frontier_response"]}</div>', unsafe_allow_html=True)
                if row.get("judge_reasoning"):
                    st.caption(f"Judge: {row['judge_reasoning']}")

        st.download_button(
            "⬇️ Download raw results (JSON)",
            data=json.dumps(results, indent=2),
            file_name="eval_results.json",
            mime="application/json"
        )
