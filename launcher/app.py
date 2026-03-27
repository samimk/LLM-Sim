"""LLM-Sim Launcher — Streamlit GUI for LLM-driven power grid optimization."""

import streamlit as st
import time
from pathlib import Path

import plotly.graph_objects as go

try:
    from config_builder import (
        scan_data_files, load_example_goals, build_config_overrides,
        get_default_config_path, DEFAULT_MODELS, BACKENDS, APPLICATIONS,
        FUTURE_APPLICATIONS, MODES,
    )
    from session_manager import SessionManager
except ModuleNotFoundError:
    from launcher.config_builder import (
        scan_data_files, load_example_goals, build_config_overrides,
        get_default_config_path, DEFAULT_MODELS, BACKENDS, APPLICATIONS,
        FUTURE_APPLICATIONS, MODES,
    )
    from launcher.session_manager import SessionManager

from llm_sim.parsers import parse_matpower, network_summary

# ── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="LLM-Sim Launcher",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session State Initialization ─────────────────────────────────────────────

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "search_running": False,
        "search_finished": False,
        "session_manager": None,
        "iteration_log": [],
        "current_phase": "",
        "current_iteration": 0,
        "stop_requested": False,
        "search_session": None,
        "search_error": None,
        "summary_analysis": None,
        "base_opflow": None,
        "best_opflow": None,
        "completed_sessions": [],
        "selected_goal_label": "Custom",
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


init_session_state()


# ── Network Info Caching ─────────────────────────────────────────────────────

@st.cache_data
def get_network_info(file_path: str) -> dict:
    """Parse a MATPOWER file and return summary info."""
    net = parse_matpower(Path(file_path))
    return {
        "buses": len(net.buses),
        "generators": len(net.generators),
        "branches": len(net.branches),
        "total_load_mw": sum(b.Pd for b in net.buses),
        "total_gen_capacity_mw": sum(g.Pmax for g in net.generators),
        "summary_text": network_summary(net),
    }


# ── Start Search Logic ───────────────────────────────────────────────────────

def start_search(base_case_path, goal, backend, model, temperature,
                 application, mode, max_iterations):
    """Initialize and start a new search."""
    overrides = build_config_overrides(
        base_case=str(base_case_path),
        backend=backend,
        model=model,
        temperature=temperature,
        application=application,
        default_mode=mode,
        max_iterations=max_iterations,
    )

    if st.session_state.session_manager is None:
        st.session_state.session_manager = SessionManager()

    manager = st.session_state.session_manager

    # Clear previous results
    st.session_state.iteration_log = []
    st.session_state.current_phase = ""
    st.session_state.current_iteration = 0
    st.session_state.search_finished = False
    st.session_state.search_session = None
    st.session_state.search_error = None
    st.session_state.summary_analysis = None
    st.session_state.base_opflow = None
    st.session_state.best_opflow = None
    st.session_state.stop_requested = False

    try:
        manager.start_search(config_overrides=overrides, goal=goal)
        st.session_state.search_running = True
    except Exception as e:
        st.error(f"Failed to start search: {e}")


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render the sidebar configuration panel and return config values."""
    disabled = st.session_state.search_running

    with st.sidebar:
        st.title("⚡ LLM-Sim")
        st.caption("v0.1.0 — LLM-Driven Power Grid Optimization")

        # ── Base Case ────────────────────────────────────────────────────
        st.header("📁 Base Case")
        data_files = scan_data_files()

        if not data_files:
            st.warning("No .m files found in data/ directory")
            selected_file = None
        else:
            display_names = [f.name for f in data_files]
            selected_idx = st.selectbox(
                "Base case file",
                range(len(data_files)),
                format_func=lambda i: display_names[i],
                disabled=disabled,
            )
            selected_file = data_files[selected_idx]
            st.caption(f"`{selected_file}`")

        # ── LLM Backend ─────────────────────────────────────────────────
        st.header("🤖 LLM Backend")
        backend = st.selectbox("Backend", BACKENDS, disabled=disabled)
        model_key = f"model_input_{backend}"
        model = st.text_input(
            "Model",
            value=DEFAULT_MODELS.get(backend, ""),
            key=model_key,
            disabled=disabled,
        )
        temperature = st.slider(
            "Temperature", 0.0, 1.0, 0.3, step=0.05,
            disabled=disabled,
        )

        # ── Search Parameters ────────────────────────────────────────────
        st.header("⚙️ Search Parameters")
        application = st.selectbox(
            "Application", APPLICATIONS, disabled=disabled,
        )
        if FUTURE_APPLICATIONS:
            st.caption(f"Coming soon: {', '.join(FUTURE_APPLICATIONS)}")

        mode = st.selectbox("Mode", MODES, disabled=disabled)
        max_iterations = st.slider(
            "Max iterations", 1, 50, 20, disabled=disabled,
        )

        # ── Search Goal ──────────────────────────────────────────────────
        st.header("🎯 Search Goal")
        example_goals = load_example_goals()
        goal_labels = ["Custom"] + [g["label"] for g in example_goals]

        selected_goal_label = st.selectbox(
            "Example Goals",
            goal_labels,
            index=goal_labels.index(st.session_state.selected_goal_label)
            if st.session_state.selected_goal_label in goal_labels
            else 0,
            disabled=disabled,
        )
        st.session_state.selected_goal_label = selected_goal_label

        # Determine default goal text
        goal_default = ""
        if selected_goal_label != "Custom":
            for g in example_goals:
                if g["label"] == selected_goal_label:
                    goal_default = g["goal"]
                    break

        goal = st.text_area(
            "Goal",
            value=goal_default,
            height=150,
            disabled=disabled,
        )

        # ── Actions ──────────────────────────────────────────────────────
        st.header("🚀 Actions")

        can_start = (
            not disabled
            and selected_file is not None
            and goal.strip() != ""
        )

        if st.button("▶️ Start Search", disabled=not can_start, type="primary"):
            start_search(
                base_case_path=selected_file,
                goal=goal.strip(),
                backend=backend,
                model=model,
                temperature=temperature,
                application=application,
                mode=mode,
                max_iterations=max_iterations,
            )
            st.rerun()

        if disabled:
            if st.button("⏹️ Stop Search", type="secondary"):
                manager = st.session_state.session_manager
                if manager is not None:
                    manager.stop_search()
                st.session_state.stop_requested = True

    return {
        "base_case": selected_file,
        "backend": backend,
        "model": model,
        "temperature": temperature,
        "application": application,
        "mode": mode,
        "max_iterations": max_iterations,
        "goal": goal,
    }


# ── Main Area — State Transitions ────────────────────────────────────────────

def render_main_area(config: dict):
    """Route to the appropriate view based on current state."""
    if st.session_state.search_running:
        render_live_monitor()
    elif st.session_state.search_finished:
        render_results()
    else:
        render_welcome(config.get("base_case"))


# ── State A: Welcome / Pre-Search ────────────────────────────────────────────

def render_welcome(base_case: Path | None):
    """Render the welcome screen before any search starts."""
    st.title("⚡ LLM-Sim — LLM-Driven Power Grid Optimization")
    st.markdown(
        "LLM-Sim uses large language models to iteratively explore and optimize "
        "power grid configurations via ExaGO simulations. It translates natural-language "
        "goals into concrete parameter modifications, runs simulations, and learns from "
        "the results to converge toward an optimal solution."
    )

    st.info(
        "Select a base case in the sidebar, choose your LLM backend, "
        "write a search goal, and click **Start Search**."
    )

    # Network summary card if a base case is selected
    if base_case is not None:
        try:
            info = get_network_info(str(base_case))
            st.subheader("📊 Network Summary")
            st.caption(f"File: `{base_case.name}`")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Buses", info["buses"])
            c2.metric("Generators", info["generators"])
            c3.metric("Branches", info["branches"])
            c4.metric("Total Load", f"{info['total_load_mw']:.1f} MW")
            c5.metric("Gen Capacity", f"{info['total_gen_capacity_mw']:.1f} MW")
        except Exception as e:
            st.error(f"Failed to parse base case: {e}")


# ── State B: Live Monitor ─────────────────────────────────────────────────────

def _iteration_icon(entry: dict) -> str:
    """Return a status icon for an iteration log entry."""
    if entry.get("convergence_status") == "FAILED" or entry.get("status") == "FAILED":
        return "❌"
    if entry.get("feasible"):
        return "✅"
    return "⚠️"


def _format_command(cmd: dict) -> str:
    """Format a single command dict as a compact summary line."""
    action = cmd.get("action", "unknown")
    parts = [f"{action}:"]
    for k, v in cmd.items():
        if k == "action":
            continue
        parts.append(f"{k}={v}")
    return " ".join(parts)


def render_live_monitor():
    """Render the live search monitor with iteration timeline and charts."""

    # 1. Poll for updates
    manager = st.session_state.session_manager
    if manager is not None:
        updates = manager.poll_updates()
        for update in updates:
            if update["type"] == "search_finished":
                st.session_state.search_running = False
                st.session_state.search_finished = True
                st.session_state.search_session = manager.get_session()
                st.session_state.search_error = manager.get_error()
                st.session_state.base_opflow = manager.get_base_opflow()
                st.session_state.best_opflow = manager.get_best_opflow()
                st.rerun()
                return
            elif update["type"] == "iteration":
                st.session_state.iteration_log.append(update)
                st.session_state.current_iteration = update["iteration"]
            elif update["type"] == "phase":
                phase_labels = {
                    "llm_request": "Sending prompt to LLM...",
                    "applying_commands": "Applying modifications...",
                    "running_simulation": "Running simulation...",
                    "parsing_results": "Parsing results...",
                }
                st.session_state.current_phase = phase_labels.get(
                    update["phase"], update["phase"]
                )
            elif update["type"] == "error":
                st.session_state.search_error = update["message"]

    # 2. Header
    st.header("🔄 Search in Progress...")

    # 3. Two-column layout
    left_col, right_col = st.columns([2, 1])

    # ── Left Column: Iteration Timeline ──────────────────────────────────
    with left_col:
        for entry in st.session_state.iteration_log:
            icon = _iteration_icon(entry)
            obj = entry.get("objective_value")
            obj_str = f"${obj:,.2f}" if obj is not None else "FAILED"
            elapsed = entry.get("sim_elapsed")
            time_str = f"{elapsed:.1f}s" if elapsed is not None else "—"
            label = (
                f"{icon} Iteration {entry['iteration']}: "
                f"{entry['description']} — {obj_str} ({time_str})"
            )

            with st.expander(label):
                # LLM Reasoning
                reasoning = entry.get("llm_reasoning")
                if reasoning:
                    st.markdown(f"*{reasoning}*")

                # Commands
                commands = entry.get("commands", [])
                if commands:
                    st.markdown(f"**Commands** ({len(commands)}):")
                    cmd_lines = [_format_command(c) for c in commands]
                    st.code("\n".join(cmd_lines), language=None)

                # Key Metrics
                v_min, v_max = entry.get("voltage_range", (0, 0))
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Voltage", f"{v_min:.3f} – {v_max:.3f} p.u."
                           if v_min > 0 else "—")
                mc2.metric("Max Line Load", f"{entry.get('max_line_loading_pct', 0):.1f}%")
                mc3.metric("Generation", f"{entry.get('total_gen_mw', 0):.1f} MW")
                mc4.metric("Violations", entry.get("violations_count", 0))

                # Mode
                st.caption(f"Mode: {entry.get('mode', '—')}")

        # Current phase indicator
        if st.session_state.search_running and st.session_state.current_phase:
            with st.status(
                f"Iteration {st.session_state.current_iteration}: "
                f"{st.session_state.current_phase}",
                state="running",
            ):
                st.write("Waiting for results...")

    # ── Right Column: Live Charts and Stats ──────────────────────────────
    with right_col:
        # Convergence Chart
        iterations = []
        values = []
        colors = []
        for entry in st.session_state.iteration_log:
            if entry["objective_value"] is not None:
                iterations.append(entry["iteration"])
                values.append(entry["objective_value"])
                colors.append("green" if entry["feasible"] else "red")

        fig = go.Figure()
        if iterations:
            fig.add_trace(go.Scatter(
                x=iterations, y=values,
                mode="lines+markers",
                marker=dict(color=colors, size=8),
                line=dict(color="rgba(100,100,100,0.5)"),
                hovertemplate="Iter %{x}<br>$%{y:,.2f}<extra></extra>",
            ))
        fig.update_layout(
            title="Convergence",
            xaxis_title="Iteration",
            yaxis_title="Objective Value ($)",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Voltage Range Chart
        iters_v = [e["iteration"] for e in st.session_state.iteration_log
                    if e["voltage_range"][0] > 0]
        v_min_list = [e["voltage_range"][0] for e in st.session_state.iteration_log
                      if e["voltage_range"][0] > 0]
        v_max_list = [e["voltage_range"][1] for e in st.session_state.iteration_log
                      if e["voltage_range"][0] > 0]

        fig2 = go.Figure()
        if iters_v:
            fig2.add_trace(go.Scatter(
                x=iters_v + iters_v[::-1],
                y=v_max_list + v_min_list[::-1],
                fill="toself",
                fillcolor="rgba(66, 133, 244, 0.2)",
                line=dict(color="rgba(0,0,0,0)"),
                name="V range",
            ))
            fig2.add_trace(go.Scatter(
                x=iters_v, y=v_max_list, mode="lines",
                line=dict(color="blue"), name="V_max",
            ))
            fig2.add_trace(go.Scatter(
                x=iters_v, y=v_min_list, mode="lines",
                line=dict(color="blue"), name="V_min",
            ))
        fig2.add_hline(y=0.95, line_dash="dash", line_color="red",
                       annotation_text="0.95 p.u.")
        fig2.add_hline(y=1.05, line_dash="dash", line_color="red",
                       annotation_text="1.05 p.u.")
        fig2.update_layout(
            title="Voltage Range",
            xaxis_title="Iteration",
            yaxis_title="Voltage (p.u.)",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Progress Stats
        st.markdown("---")
        n_iters = len(st.session_state.iteration_log)
        feasible_count = sum(
            1 for e in st.session_state.iteration_log if e.get("feasible")
        )
        st.metric("Iterations", n_iters)
        st.metric("Feasible", feasible_count)
        if st.session_state.iteration_log:
            feasible_entries = [
                e for e in st.session_state.iteration_log
                if e.get("feasible") and e.get("objective_value") is not None
            ]
            if feasible_entries:
                best = min(feasible_entries, key=lambda e: e["objective_value"])
                st.metric(
                    "Best Cost",
                    f"${best['objective_value']:,.2f}",
                    delta=f"Iter {best['iteration']}",
                )

    # 4. Auto-rerun (MUST be last)
    if st.session_state.search_running:
        time.sleep(1)
        st.rerun()


# ── State C: Results (Skeleton) ──────────────────────────────────────────────

def render_results():
    """Render results view. Full implementation in Step 7."""
    st.header("✅ Search Complete")

    session = st.session_state.search_session
    if session is None:
        if st.session_state.search_error:
            st.error(f"Search failed: {st.session_state.search_error}")
        else:
            st.warning("No session data available.")
        return

    st.write(f"**Goal:** {session.goal}")
    st.write(f"**Termination:** {session.termination_reason}")
    stats = session.journal.summary_stats()
    st.write(f"**Iterations:** {stats['total_iterations']}")
    if stats["best_objective"] is not None:
        st.write(
            f"**Best objective:** ${stats['best_objective']:,.2f} "
            f"(iteration {stats['best_iteration']})"
        )

    st.info("📊 Full results visualization coming in Step 7.")

    if st.button("🔄 Start New Search"):
        st.session_state.search_finished = False
        st.session_state.search_session = None
        st.rerun()


# ── Main Entry Point ─────────────────────────────────────────────────────────

config = render_sidebar()
render_main_area(config)
