"""Live Streamlit UI for sys1 — watch the logit-based judgment engine work.

    streamlit run ui/app.py
"""
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sys1 import Choice, Noul, Score  # noqa: E402
from sys1.engine import Engine, DEFAULT_MODEL  # noqa: E402

st.set_page_config(page_title="sys1", page_icon="\U0001f9e0", layout="centered")


@st.cache_resource(show_spinner="Loading base model (first run downloads it)...")
def load_engine(model_name: str) -> Engine:
    return Engine(model_name=model_name)


st.title("sys1")
st.caption(
    "A tiny 'System One' model: typed judgments with real probabilities read "
    "straight from next-token logits, not text the model guesses at."
)

with st.sidebar:
    st.header("Model")
    model_name = st.text_input("Base model", value=DEFAULT_MODEL)
    st.caption("Any small HF instruct model works. First load downloads and caches it.")
    primitive = st.radio("Primitive", ["Choice", "Noul", "Score"])

engine = load_engine(model_name)

instructions = st.text_area(
    "Instructions",
    value="Classify what this support ticket is about.",
    height=80,
)
state_text = st.text_area(
    "State (JSON, optional)",
    value='{"ticket": "My card was charged twice for the same order, please refund me."}',
    height=100,
)

state = None
if state_text.strip():
    try:
        state = json.loads(state_text)
    except json.JSONDecodeError as e:
        st.error(f"State must be valid JSON: {e}")
        st.stop()

if primitive == "Choice":
    options_text = st.text_area(
        "Options (one per line)",
        value="billing\ntechnical support\nsales\nspam",
        height=100,
    )
    options = [line.strip() for line in options_text.splitlines() if line.strip()]
elif primitive == "Score":
    levels_text = st.text_area(
        "Levels, low to high (one per line)",
        value="not urgent\nsomewhat urgent\nurgent\ncritical",
        height=100,
    )
    levels = [line.strip() for line in levels_text.splitlines() if line.strip()]

run = st.button("Run judgment", type="primary")

if run:
    if primitive == "Choice":
        result = Choice(options).ask(instructions, state, engine)
        st.subheader(f"Choice: **{result['choice']}**")
        st.bar_chart(result["probabilities"])
        st.json(result["probabilities"])
    elif primitive == "Noul":
        result = Noul().ask(instructions, state, engine)
        p_yes = result["probability_yes"]
        st.subheader(f"Probability yes: **{p_yes:.1%}**")
        st.progress(p_yes)
        st.bar_chart({"yes": p_yes, "no": 1 - p_yes})
    else:
        result = Score(levels).ask(instructions, state, engine)
        st.subheader(f"Level: **{result['level']}** (expected index {result['expected_index']:.2f})")
        st.bar_chart(result["probabilities"])
        st.json(result["probabilities"])
