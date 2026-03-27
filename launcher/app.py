"""LLM-Sim Launcher — Streamlit GUI for LLM-driven power grid optimization."""

import streamlit as st
import time
from pathlib import Path

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
        model = st.text_input(
            "Model",
            value=DEFAULT_MODELS.get(backend, ""),
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


# ── State B: Live Monitor (Skeleton) ─────────────────────────────────────────

def render_live_monitor():
    """Render the live search monitor. Full implementation in Step 5."""
    st.header("🔄 Search in Progress...")

    manager = st.session_state.session_manager
    if manager is not None:
        updates = manager.poll_updates()
        for update in updates:
            if update["type"] == "iteration":
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
            elif update["type"] == "search_finished":
                st.session_state.search_running = False
                st.session_state.search_finished = True
                st.session_state.search_session = manager.get_session()
                st.session_state.search_error = manager.get_error()
                st.session_state.base_opflow = manager.get_base_opflow()
                st.session_state.best_opflow = manager.get_best_opflow()
                st.rerun()
            elif update["type"] == "error":
                st.session_state.search_error = update["message"]

    # Show current status
    if st.session_state.current_phase:
        st.info(
            f"Iteration {st.session_state.current_iteration}: "
            f"{st.session_state.current_phase}"
        )

    # Show iteration log (simple version — will be replaced in Step 5)
    for entry in st.session_state.iteration_log:
        status_icon = "✅" if entry.get("feasible") else "❌"
        obj = entry.get("objective_value")
        obj_str = f"${obj:,.2f}" if obj is not None else "FAILED"
        st.write(
            f"{status_icon} **Iter {entry['iteration']}**: "
            f"{entry['description']} — {obj_str}"
        )

    # Auto-rerun to poll for updates
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
