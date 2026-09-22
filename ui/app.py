"""Live Streamlit UI for sys1 — watch the logit-based judgment engine work.

    streamlit run ui/app.py
"""
import json
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sys1 import Choice, Noul, Score  # noqa: E402
from sys1.engine import Engine, DEFAULT_MODEL  # noqa: E402
from sys1.games import SnakeGame, decide_move  # noqa: E402

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
    default_model = (
        "models/sys1-calibrated-out"
        if Path("models/sys1-calibrated-out").is_dir()
        else "models/sys1-lora-out"
        if Path("models/sys1-lora-out").is_dir()
        else DEFAULT_MODEL
    )
    model_name = st.text_input("Model (base HF id or local adapter path)", value=default_model)
    st.caption("Any small HF instruct model works, or a local LoRA adapter directory.")
    mode = st.radio("Mode", ["Judgment playground", "Snake game"])

engine = load_engine(model_name)

if mode == "Judgment playground":
    primitive = st.radio("Primitive", ["Choice", "Noul", "Score"])

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

else:
    st.markdown(
        "The game engine owns every rule, including safety: it already knows "
        "which directions are safe (no wall, no self-collision) and only "
        "asks the model when there's a real choice among *safe* directions. "
        "The model's only job is picking the safe direction that heads "
        "toward the food — it is never asked to (re-)derive safety itself."
    )

    st.caption("Defaults (24x16, length 6) match the laya-mlx snake demo for a fair comparison.")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        grid_w = st.slider("Grid width", 6, 32, 24)
    with col2:
        grid_h = st.slider("Grid height", 6, 32, 16)
    with col3:
        initial_length = st.slider("Initial length", 3, 10, 6)
    with col4:
        max_ticks = st.slider("Max ticks", 20, 2000, 800)

    speed = st.slider("Seconds per tick (animation speed)", 0.0, 0.5, 0.04)
    play = st.button("Play", type="primary")

    grid_box = st.empty()
    info_box = st.empty()

    def render_grid(game: SnakeGame) -> str:
        grid = [["." for _ in range(game.width)] for _ in range(game.height)]
        for x, y in game.body[1:]:
            grid[y][x] = "o"
        hx, hy = game.head()
        grid[hy][hx] = "H"
        fx, fy = game.food
        grid[fy][fx] = "F"
        return "\n".join(" ".join(row) for row in grid)

    if play:
        game = SnakeGame(width=grid_w, height=grid_h, initial_length=initial_length)
        grid_box.code(render_grid(game), language=None)
        info_box.write(f"tick 0 | score 0 | alive")

        def choose(instructions, options, state):
            return Choice(options).ask(instructions, state, engine)

        for _ in range(max_ticks):
            direction, judgment = decide_move(game, choose)
            alive, ate = game.step(direction)

            grid_box.code(render_grid(game), language=None)
            status = "alive" if alive else "DIED"
            if judgment is not None:
                probs = ", ".join(f"{k}={v:.2f}" for k, v in judgment["probabilities"].items())
                decision_line = f"chose **{direction}** ({probs})"
            else:
                decision_line = f"forced move: **{direction}** (only safe option, no judgment needed)"
            info_box.write(f"tick {game.ticks} | score {game.score} | {status}  \n{decision_line}")

            if speed > 0:
                time.sleep(speed)
            if not alive:
                break

        st.success(f"Game over — survived {game.ticks} ticks, score {game.score}")
