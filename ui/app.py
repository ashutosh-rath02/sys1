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
    mode = st.radio("Mode", ["Judgment playground", "Snake game"])
    if mode == "Snake game":
        # 0.5B for speed: each tick is a live forward pass on CPU, and the
        # 1.5B model's per-tick latency makes live play unwatchably slow.
        default_model = (
            "models/sys1-lora-out" if Path("models/sys1-lora-out").is_dir() else DEFAULT_MODEL
        )
    else:
        default_model = (
            "models/sys1-calibrated-out"
            if Path("models/sys1-calibrated-out").is_dir()
            else "models/sys1-lora-out"
            if Path("models/sys1-lora-out").is_dir()
            else DEFAULT_MODEL
        )
    model_name = st.text_input(
        "Model (base HF id or local adapter path)", value=default_model, key=f"model_input_{mode}"
    )
    st.caption("Any small HF instruct model works, or a local LoRA adapter directory.")

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
        "Memory lives in the game engine, not the model: once it commits to "
        "closing the gap on one axis (x or y), following that plan is fully "
        "deterministic code, persisted in `game.focus_axis` across ticks — "
        "no judgment, and nothing for the model to forget. The model is "
        "asked only for the two things that are genuinely ambiguous: which "
        "axis to tackle first when both are open, and what to do when the "
        "planned move turns out to be unsafe. Most ticks make zero model calls "
        "and run instantly; a real forward pass only happens on the rest — "
        "that's why play speeds up and slows down instead of ticking evenly."
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

    speed = st.slider("Seconds per deterministic tick (animation pacing)", 0.0, 0.3, 0.03)
    play = st.button("Play", type="primary")

    CELL_PX = 16
    COLOR_EMPTY = "#1c1f26"
    COLOR_BODY = "#35c9c9"
    COLOR_HEAD = "#f5f5f5"
    COLOR_FOOD = "#ff4d6d"
    COLOR_BG = "#0e1117"

    def render_grid_html(game: SnakeGame) -> str:
        body_set = set(game.body[1:])
        hx, hy = game.head()
        fx, fy = game.food
        cells = []
        for y in range(game.height):
            for x in range(game.width):
                if (x, y) == (hx, hy):
                    color = COLOR_HEAD
                elif (x, y) == (fx, fy):
                    color = COLOR_FOOD
                elif (x, y) in body_set:
                    color = COLOR_BODY
                else:
                    color = COLOR_EMPTY
                cells.append(
                    f'<div style="width:{CELL_PX}px;height:{CELL_PX}px;'
                    f'background:{color};border-radius:2px;"></div>'
                )
        return (
            f'<div style="display:inline-grid;grid-template-columns:repeat({game.width}, {CELL_PX}px);'
            f'gap:2px;background:{COLOR_BG};padding:10px;border-radius:8px;">'
            + "".join(cells)
            + "</div>"
        )

    def render_bars_html(probs: dict | None) -> str:
        if not probs:
            return '<div style="color:#666;font-family:monospace;">no judgment yet this game</div>'
        rows = []
        for label, p in sorted(probs.items(), key=lambda kv: -kv[1]):
            pct = round(p * 100)
            rows.append(
                '<div style="display:flex;align-items:center;gap:8px;margin:3px 0;'
                'font-family:monospace;font-size:13px;">'
                f'<div style="width:70px;color:#ccc;">{label.upper()}</div>'
                f'<div style="flex:1;background:{COLOR_EMPTY};border-radius:3px;height:14px;">'
                f'<div style="width:{pct}%;background:{COLOR_BODY};height:100%;border-radius:3px;"></div></div>'
                f'<div style="width:40px;color:#ccc;text-align:right;">{pct}%</div></div>'
            )
        return "".join(rows)

    grid_box = st.empty()
    thinking_box = st.empty()
    stats_box = st.empty()
    bars_box = st.empty()

    if play:
        game = SnakeGame(width=grid_w, height=grid_h, initial_length=initial_length)
        grid_box.markdown(render_grid_html(game), unsafe_allow_html=True)

        decision_times: list[float] = []
        last_probs: dict | None = None

        def choose(instructions, options, state):
            thinking_box.markdown("🤔 **model thinking…**")
            start = time.perf_counter()
            result = Choice(options).ask(instructions, state, engine)
            decision_times.append(time.perf_counter() - start)
            thinking_box.empty()
            return result

        n_asked = 0
        start_time = time.perf_counter()
        for tick_num in range(1, max_ticks + 1):
            direction, judgment = decide_move(game, choose)
            n_asked += judgment is not None
            if judgment is not None:
                last_probs = judgment["probabilities"]
            alive, ate = game.step(direction)

            grid_box.markdown(render_grid_html(game), unsafe_allow_html=True)
            bars_box.markdown(render_bars_html(last_probs), unsafe_allow_html=True)

            elapsed = max(time.perf_counter() - start_time, 1e-6)
            avg_decision_ms = (sum(decision_times) / len(decision_times) * 1000) if decision_times else 0.0
            with stats_box.container():
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Score", game.score)
                c2.metric("Ticks/s", f"{tick_num / elapsed:.1f}")
                c3.metric("Model calls", f"{n_asked}/{tick_num}", f"{n_asked / tick_num:.0%}")
                c4.metric("Avg decision", f"{avg_decision_ms:.0f} ms")
                c5.metric("Status", "alive" if alive else "DIED")

            if judgment is None and speed > 0:
                time.sleep(speed)
            if not alive:
                break

        st.success(f"Game over — survived {game.ticks} ticks, score {game.score}, {n_asked} model calls")
