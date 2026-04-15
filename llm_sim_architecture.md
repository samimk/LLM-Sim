# LLM-Sim — Architecture Document

**LLM-Driven Iterative Simulation and Analysis for ExaGO**

| Field | Value |
|-------|-------|
| Project | LLM-Sim (to be integrated into ExaGO) |
| Version | 0.2 — Enhanced PoC |
| Date | April 2026 |
| Supervisor | Slaven P., Oak Ridge National Laboratory |
| Target Platform | Linux (OpenSUSE Leap 15.6), Python 3.10+ |

---

## 1. Introduction and Motivation

LLM-Sim is a standalone project that brings large language model (LLM) intelligence into the simulation loop of power grid analysis. Rather than using an LLM merely as a conversational interface for setting up a single simulation run, LLM-Sim positions the LLM as an iterative decision-making agent that proposes parameter modifications, triggers simulations via ExaGO, interprets results, and decides subsequent actions — repeating this cycle until a user-defined goal is achieved.

This approach is motivated by a class of power system analysis tasks that are difficult to express as formal optimization problems but natural to describe in plain language: "find the load level at which the network becomes infeasible," "identify the weakest corridor under stress," or "find a dispatch that balances cost and voltage quality." A human expert would approach these tasks through iterative exploration — running a simulation, examining results, forming hypotheses, adjusting parameters, and repeating. LLM-Sim automates this expert workflow.

The project is developed as a self-contained codebase that will later be incorporated into the ExaGO project, a large-scale power grid optimization toolkit developed at Oak Ridge National Laboratory and designed for execution on exascale computers such as Frontier.

---

## 2. Design Goals and Principles

- **LLM-in-the-loop (Option B architecture):** The LLM is the decision-maker in the search loop — it proposes parameter changes, interprets simulation results, and decides the next action. No external optimization algorithm is interposed.
- **Application-agnostic design:** While the PoC targets OPFLOW, the architecture treats the inner ExaGO application as a pluggable component. Supporting additional applications (SCOPFLOW, TCOPFLOW, SOPFLOW, DCOPFLOW, PFLOW) requires only defining their parameter schemas and output parsers.
- **Base case + deltas model:** A user-provided base case file (.m and optionally .gic) is immutable throughout the search. Each iteration creates a working copy and applies the LLM's modifications as structured JSON commands.
- **Multi-backend LLM support:** The system supports OpenAI API, Anthropic API, and Ollama (including Ollama Cloud) through a unified abstraction layer.
- **Iterative and bounded search:** Every search session has a configurable maximum iteration count (default: 20). The LLM can terminate early by declaring the goal achieved.
- **Interactive steering:** Users can inject directives into a running search (augment or replace mode) and pause/resume at iteration boundaries without restarting.
- **Goal-type-aware reporting:** Post-search analysis classifies the search goal (cost minimization, feasibility boundary, constraint satisfaction, parameter exploration) and selects the best iteration accordingly, rather than always defaulting to lowest cost.
- **Application-aware commands:** The system warns when commands are ineffective for the current application (e.g., `set_gen_voltage` under OPFLOW) and guides the LLM toward effective alternatives.
- **Designed for future multi-objective reasoning:** The system anticipates qualitative multi-objective tasks where the LLM balances competing goals.

---

## 3. System Architecture Overview

The system is organized into three layers:

| Layer | Component | Responsibility |
|-------|-----------|---------------|
| Outer | Agent Loop Controller | Manages the iterative cycle: assembles LLM prompts, sends to LLM backend, parses responses, decides continue/stop |
| Middle | Simulation Orchestrator | Applies modification commands to working copy, invokes ExaGO, captures output, parses results |
| Inner | ExaGO Application | The actual simulation engine (OPFLOW, SCOPFLOW, etc.) |

### Data flow for one iteration:

1. Agent Loop Controller assembles prompt (goal + search journal + latest results + steering directives)
2. LLM backend generates response with structured action (modify / complete / analyze)
3. Controller parses response; if action is "modify", extracts JSON commands
4. Modification Engine validates and applies commands to a working copy of the base case; application-specific warnings are generated (e.g., `set_gen_voltage` under OPFLOW) and fed back to the LLM
5. Simulation Executor invokes ExaGO with the modified input files
6. Results Parser extracts structured data from ExaGO output and checks violations against actual bus voltage limits from the modified network
7. Controller updates the Search Journal with iteration summary
8. Loop returns to step 1 (or terminates if action is "complete" or max iterations reached)

### Post-search finalization:

9. A final LLM call classifies the search goal type and identifies the best iteration
10. Summary statistics, charts, and PDF report are generated with goal-type-aware framing

---

## 4. Component Design

### 4.1 LLM Backend Abstraction Layer

```python
class LLMBackend(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.3) -> str: ...
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def supports_json_mode(self) -> bool: ...
```

| Provider | Config Key | Recommended Models | Notes |
|----------|------------|-------------------|-------|
| OpenAI | openai | GPT-4o, GPT-4o-mini | JSON mode supported |
| Anthropic | anthropic | Claude Sonnet 4.6 | JSON via system prompt; excellent JSON reliability |
| Ollama (local) | ollama | Qwen 2.5 7B/14B, Llama 3.x | JSON mode varies |
| Ollama Cloud | ollama-cloud | GLM-5 | ~60% JSON reliability; GLM-5.1 incompatible |

Configuration via YAML file or environment variables. Temperature kept low (0.2–0.4).

**Model compatibility notes:** GLM-5 via Ollama Cloud works but has ~40% JSON parse failure rate, wasting iterations. GLM-5.1 is currently incompatible (fails to produce valid JSON actions). Claude Sonnet 4.6 via Anthropic API offers near-perfect JSON reliability at ~$0.70–0.80 per run on the ACTIVSg200 network.

### 4.2 Simulation Executor

Two execution modes:

| Mode | Description | Advantage |
|------|-------------|-----------|
| CLI subprocess | Invokes ExaGO binaries as subprocesses | Simpler setup; works without Python bindings |
| Python API | Uses ExaGO Python bindings | Direct access to solution objects |

The PoC implements CLI subprocess mode first. Each ExaGO application is registered with:

| Application | Binary | Inputs | Key Outputs |
|-------------|--------|--------|-------------|
| OPFLOW | opflow | .m [.gic] | Bus voltages, gen dispatch, line flows, cost, convergence |
| DCOPFLOW | dcopflow | .m | DC angles, active power flows, cost (same output format as OPFLOW; voltages fixed at 1.0 pu) |
| SCOPFLOW | scopflow | .m + contingency | Base case + contingency results |
| TCOPFLOW | tcopflow | .m + load profile | Multi-period dispatch |
| SOPFLOW | sopflow | .m + scenarios | Stochastic results across scenarios |
| PFLOW | pflow | .m | Power flow solution (no optimization) |

### 4.3 Results Parser

Produces two outputs:
- **Full structured result:** Python dictionary with all extracted data. Stored for detailed queries.
- **Compact summary:** 15–30 lines of text for LLM prompt inclusion. Contains: objective value, feasibility status, violation count/severity, voltage range, top-5 loaded lines, total generation vs. load.

**Application-aware dispatch:** The parser layer includes dispatch functions (`parse_simulation_result_for_app`, `results_summary_for_app`) that route to the correct parser and summary generator based on the application name. DCOPFLOW reuses the OPFLOW parser (identical output format) but has its own summary generator (`dcopflow_results_summary`) that shows phase angle profiles instead of voltage magnitudes and omits reactive power data. Unknown applications fall back to the OPFLOW parser with a logged warning.

**Violation checking** uses actual bus voltage limits (`Vmin`/`Vmax`) from the input MATPOWER network rather than hardcoded thresholds. This ensures that when the LLM tightens voltage limits via `set_bus_vlimits` or `set_all_bus_vlimits`, violations are reported accurately against the enforced limits. If bus limits are not available (backward compatibility), falls back to 0.9/1.1 p.u.

### 4.4 Search Journal

In-memory data structure — one entry per iteration:

| Field | Type | Description |
|-------|------|-------------|
| iteration | int | Sequential iteration number |
| description | str | LLM-provided description of changes |
| commands | list[dict] | JSON modification commands applied |
| objective_value | float\|None | ExaGO objective function value |
| feasible | bool | Whether simulation converged without violations |
| violations_count | int | Number of constraint violations |
| voltage_range | str | Min–max bus voltage magnitudes |
| max_line_loading | float | Highest line loading % |
| total_gen_mw | float | Total generation MW |
| total_load_mw | float | Total load MW |
| llm_reasoning | str | LLM's explanation of its choices |
| mode | str | "accumulative" or "fresh" |
| steering_directive | str\|None | Active steering directive at this iteration |

The `summary_stats()` method accepts optional `best_iteration_override` and `goal_type` parameters, allowing the post-search goal classification to select the correct best iteration rather than always defaulting to lowest cost.

### 4.5 Agent Loop Controller

Responsibilities:
- **Prompt assembly:** Combines system prompt + goal + journal + latest results + steering directives + error/warning feedback
- **Response parsing:** Extracts JSON from LLM output (handles markdown fences, extra text)
- **Action dispatch:** Routes "modify" → Modification Engine + Executor; "analyze" → detailed query; "complete" → terminate with report
- **Warning feedback:** Modifier warnings (e.g., `set_gen_voltage` ineffective under OPFLOW) are fed back to the LLM in the next iteration's prompt, allowing self-correction
- **Termination logic:** Stops on LLM completion, max iterations, or critical error (3 consecutive JSON parse failures)
- **Modification mode management:** Supports "accumulative" and "fresh" modes
- **Post-search analysis:** Makes a final LLM call to classify the search goal type and identify the best iteration; stores classification and analysis text on `SearchSession`

### 4.6 Goal Classification

After the search loop completes, a shared utility module (`llm_sim/engine/goal_classifier.py`) builds a classification prompt and parses the LLM's structured JSON response. Both the CLI path (`AgentLoopController._finalize`) and the GUI path (`SessionManager.get_summary_analysis`) use this shared logic.

Goal types:

| Goal Type | Description | Best Iteration Selection |
|-----------|-------------|------------------------|
| `cost_minimization` | Minimize generation cost | Lowest cost among feasible |
| `feasibility_boundary` | Find parameter limit before infeasibility | Feasible iteration closest to boundary |
| `constraint_satisfaction` | Satisfy specific constraints | Iteration best satisfying constraints |
| `parameter_exploration` | Explore what-if scenarios | Most informative feasible iteration |

If classification fails, falls back to `cost_minimization` (lowest cost heuristic).

### 4.7 Modification Engine

Deterministic Python module that applies JSON commands to MATPOWER .m files:

| Command | Parameters | Description |
|---------|-----------|-------------|
| `set_load` | {bus, Pd, Qd} | Set load at a specific bus |
| `scale_load` | {area\|zone\|bus, factor} | Scale load by factor |
| `scale_all_loads` | {factor} | Scale all loads uniformly |
| `set_gen_status` | {bus, id?, status} | Turn generator on/off |
| `set_gen_dispatch` | {bus, id?, Pg} | Set generator power output |
| `set_gen_voltage` | {bus, id?, Vg} | Set generator voltage setpoint (initial guess only — see note below) |
| `set_branch_status` | {fbus, tbus, ckt?, status} | Enable/disable branch |
| `set_branch_rate` | {fbus, tbus, ckt?, rateA} | Modify thermal rating |
| `set_cost_coeffs` | {bus, id?, coeffs} | Modify cost curve |
| `set_bus_vlimits` | {bus, Vmin, Vmax} | Set voltage bounds on a single bus |
| `set_all_bus_vlimits` | {Vmin, Vmax} | Set voltage bounds on ALL buses at once |
| `set_ground_resistance` | {substation, R_ground} | Modify grounding (GIC) |

All commands validated before application. Invalid commands reported to LLM as errors. Application-specific warnings are generated and fed back to the LLM.

**OPF voltage control note:** In OPFLOW, bus voltages are optimization decision variables — the solver picks optimal voltages within `Vmin`/`Vmax` bounds. The `set_gen_voltage` command only sets the initial guess (`Vg`), which OPFLOW overrides. To actually constrain voltages in OPF, use `set_all_bus_vlimits` (system-wide) or `set_bus_vlimits` (per-bus). The system prompt and command schema include this guidance, and a runtime warning is emitted when `set_gen_voltage` is used under OPFLOW.

### 4.8 User Interfaces

**CLI:** User provides base case path, application, search goal, optional config. Progress displayed live; post-search analysis with goal classification; summary report on completion.

**Streamlit GUI Launcher (`launcher/`):** Three-state application:
- **State A (Welcome):** Network summary, configuration sidebar, search goal input with preset examples
- **State B (Live Monitor):** Real-time iteration timeline with expandable details, convergence and voltage range charts, steering panel (augment/replace/pause/resume)
- **State C (Results):** Three tabs — Overview (summary metrics, comparison table, convergence charts), Detailed Results (voltage profile, generator dispatch, line loading charts, iteration history), Analysis & Report (LLM analysis with goal classification, search narrative, PDF report generation)

The `SessionManager` bridges the Streamlit UI with `AgentLoopController` via a background thread with callback-based updates through a thread-safe queue.

### 4.9 PDF Report Generator

Generates comprehensive PDF reports using ReportLab with DejaVu Sans font for diacritics support. Report sections:

1. **Title page** — search type, goal, metadata
2. **Executive summary** — goal-type-aware framing of results (cost reduction for cost minimization, neutral framing for feasibility boundary searches), LLM analysis with markdown table rendering
3. **Convergence analysis** — objective value and voltage range charts
4. **Results comparison** — base case vs best solution table and charts (voltage profile, generator dispatch)
5. **Iteration log** — complete search history
6. **Steering history** — user directives injected during the search

---

## 5. Data Flow and Iteration Lifecycle

1. **Prompt Construction** — Controller reads journal, formats compact table, combines with system prompt (command schema + network metadata + OPF-specific guidance), goal statement, steering directives, and previous results summary. Warning/error feedback from prior iteration is included.
2. **LLM Inference** — Prompt sent to selected backend. Expected response: JSON with "action" (modify|complete|analyze), "reasoning", and action-specific payload.
3. **Response Parsing** — JSON extracted, handling markdown fences and extra text. Parse failure → error logged, error feedback sent to LLM, loop continues (up to 3 consecutive failures).
4. **Command Validation** — Each command validated against network data. Invalid → error message fed back to LLM. Application-specific warnings generated (e.g., `set_gen_voltage` under OPFLOW) and also fed back.
5. **File Modification** — Valid commands applied to working copy. Written to timestamped directory.
6. **Simulation Execution** — ExaGO invoked. Timeout guard (default 120s).
7. **Results Parsing** — Full structured result + compact summary produced. Violations checked against actual bus Vmin/Vmax from the modified network.
8. **Journal Update** — New entry appended with iteration metadata and results.
9. **Loop Decision** — Continue unless LLM declared complete or max iterations reached.
10. **Post-Search Finalization** — Final LLM call for goal classification and analytical summary. Results stored on `SearchSession` for both CLI display and GUI reuse.

---

## 6. LLM Prompt Architecture

Four sections assembled per iteration:

**Section A — System Prompt (static):** Role definition, command schema with parameter types and validation rules (including OPF voltage control guidance), network metadata (bus count, generators with fuel types and capacities, areas, voltage levels), response format specification.

**Section B — Goal Statement (static):** User's original natural language goal, preserved exactly.

**Section C — Search Journal (grows):** Compact table, e.g.:
```
Iter | Description                      | Cost($)   | Feasible | V_range(pu)  | Max_load(%)
-----|----------------------------------|-----------|----------|--------------|------------
  1  | Base case (no modifications)     | 27,361.42 | Yes      | 0.95 – 1.05  | 78.3%
  2  | Scale all loads +10%             | 30,154.88 | Yes      | 0.94 – 1.06  | 86.1%
  3  | Scale all loads +20%             | 33,019.55 | Yes      | 0.92 – 1.07  | 93.7%
```

**Section D — Latest Results (refreshed):** Compact summary from most recent simulation.

**Section E — Steering Directives (when active):** Operator directives injected via augment/replace mode.

**Section F — Error/Warning Feedback (when present):** Command errors, parse failures, and application-specific warnings from the previous iteration.

### Expected LLM response formats:

Modify action:
```json
{
  "action": "modify",
  "reasoning": "Loads at +20% pushed voltages near limits. Trying +25% to find boundary.",
  "mode": "fresh",
  "commands": [
    {"action": "scale_all_loads", "factor": 1.25}
  ]
}
```

Complete action:
```json
{
  "action": "complete",
  "reasoning": "Network becomes infeasible between +25% and +30% load scaling.",
  "findings": {
    "summary": "The maximum feasible uniform load increase is approximately 27%.",
    "critical_buses": [42, 87, 153],
    "critical_lines": [{"from": 42, "to": 87, "loading_pct": 99.1}],
    "recommendation": "Reinforce the 230kV corridor between buses 42-87."
  }
}
```

---

## 7. Search Modes and Use Cases

| Mode | Name | Example Goal | Search Pattern | Phase |
|------|------|-------------|---------------|-------|
| A | Scenario Exploration | What happens if load +10% while largest gen trips? | Sequential, few iterations | 1 |
| B | Sensitivity / Boundary Finding | At what load level does network become infeasible? | Binary search / bisection | 1 |
| C | Qualitative Multi-Objective | Find cheap dispatch with healthy voltages | Iterative refinement with tradeoffs | 2 |
| D | Adversarial / Stress Testing | Find worst-case N-2 contingency | Combinatorial exploration | 3 |

**Accumulative mode:** Each iteration builds on previous modified state. Natural for incremental tuning.
**Fresh-start mode:** Each iteration starts from base case. Natural for comparative exploration.

---

## 8. Interactive Steering (Implemented)

Users can inject steering directives into a running search without stopping it.

### Mechanism

- **Steering queue** (`queue.Queue`): Any thread (CLI listener, GUI) calls `controller.inject_steering(directive, mode)`. Items are drained at the start of each iteration boundary.
- **Directive modes:**
  - `augment` — the directive is added alongside the current goal. Multiple augment directives accumulate.
  - `replace` — the directive replaces all previously active directives (clears the accumulation list), then becomes the sole active directive.
- **Active directives** are injected into the user prompt as an `=== Operator Directives ===` section. The LLM is instructed on how each mode affects priority.
- **Steering history** is recorded per iteration in `JournalEntry.steering_directive` and in `AgentLoopController._steering_history`. It is included in the PDF report.

### Pause / Resume

- `controller.pause()` — clears `threading.Event._pause_event`; the search thread calls `_pause_event.wait()` at the next iteration boundary and blocks without busy-waiting.
- `controller.resume()` — sets the event, unblocking the thread.
- An optional `on_pause_state(paused: bool)` callback notifies the GUI.

### CLI entry point

`llm_sim/cli.py` starts a daemon thread (`_start_stdin_listener`) when stdin is a TTY. Accepted commands: `pause`, `resume`, `stop`, `status`, `replace: <text>`, or bare text (augment).

### GUI entry point

The Streamlit launcher's live monitor exposes a steering text input plus Augment, Replace, and Pause/Resume buttons that forward to `SessionManager`, which delegates to `AgentLoopController`.

### Branching (Future)

Saving a state snapshot and exploring alternative search paths from a prior iteration is not yet implemented.

---

## 9. Tested Configurations and Results

### ACTIVSg200 — Maximum Loadability Search

| Backend | Model | Result | JSON Failures | Duration | Notes |
|---------|-------|--------|---------------|----------|-------|
| ollama-cloud | GLM-5 | λ=1.462× (90.0% loading) | ~40% | 158s (17 iter) | Best result with `set_all_bus_vlimits` |
| ollama-cloud | GLM-5 | Stuck at voltage issue | ~40% | 947s (15 iter) | Before `set_all_bus_vlimits` — LLM issued 10/200 bus limits |
| ollama-cloud | GLM-5.1 | No iterations executed | 100% | 1090s | Incompatible JSON output format |
| anthropic | Claude Sonnet 4.6 | Not yet tested | Expected ~0% | Expected ~$0.70 | Planned |

Key finding: the `set_all_bus_vlimits` command was essential for this test case — without it, the LLM could only set voltage limits on a handful of buses per iteration due to JSON output size constraints, leaving the majority of buses unconstrained.

---

## 10. Multi-Objective Decision Making (Implemented)

- **Tradeoff articulation:** LLM explains cost of moving between solutions in natural language.
- **Pareto approximation:** Systematic exploration presenting non-dominated solutions.
- **Preference elicitation:** Clarifying questions about user priorities via interactive steering.
- **Constraint softening:** Distinguishing hard constraints from soft preferences.

---

## 11. Phased Implementation Plan

### Phase 1 — Foundation + OPFLOW PoC ✅

| Step | Task | Details | Status |
|------|------|---------|--------|
| 1.1 | Project scaffolding | Directory structure, config (YAML), CLI args, logging | ✅ |
| 1.2 | LLM backend abstraction | Base class + OpenAI, Anthropic, Ollama adapters | ✅ |
| 1.3 | MATPOWER file parser/writer | Read .m → Python structures; write back preserving format | ✅ |
| 1.4 | Modification engine | Apply JSON commands; validate; report errors; application-specific warnings | ✅ |
| 1.5 | Simulation executor (CLI) | Invoke opflow binary, capture output, handle timeouts | ✅ |
| 1.6 | OPFLOW results parser | Extract objective, voltages, dispatch, flows, convergence; violation checking against actual bus limits | ✅ |
| 1.7 | Search journal | In-memory structure with append, format, export; goal-type-aware summary stats | ✅ |
| 1.8 | Agent loop controller | Prompt assembly, response parsing, action dispatch, termination, post-search goal classification | ✅ |
| 1.9 | System prompt templates | OPFLOW prompts with command schema, network metadata, OPF voltage control guidance | ✅ |
| 1.10 | CLI interface | Goal input, live progress, goal-type-aware final report | ✅ |
| 1.11 | End-to-end testing | ACTIVSg200 case, Ollama Cloud (GLM-5), feasibility boundary + steering | ✅ |

### Phase 1+ — Launcher + Reporting ✅

| Step | Task | Details | Status |
|------|------|---------|--------|
| 1+.1 | Streamlit GUI launcher | Config builder, session manager, live monitor, results view | ✅ |
| 1+.2 | PDF report generator | ReportLab-based, DejaVu Sans, markdown table rendering | ✅ |
| 1+.3 | Goal classification | Shared post-search LLM analysis for CLI and GUI paths | ✅ |
| 1+.4 | Goal-type-aware reporting | Correct framing for feasibility boundary, constraint satisfaction, etc. | ✅ |
| 1+.5 | Interactive steering (GUI) | Augment/replace directives, pause/resume, steering history | ✅ |
| 1+.6 | Bulk voltage commands | `set_all_bus_vlimits` for system-wide voltage constraint enforcement | ✅ |

### Phase 2 — Enrichment + Multi-Objective

| Step | Task | Details |
|------|------|---------|
| 2.1 | Interactive steering | ✅ Implemented |
| 2.2 | Analyze action | ✅ Implemented (pattern-matched + LLM fallback) |
| 2.3 | Multi-objective tracking | ✅ Implemented (objective registry, 14 metric extractors, LLM-driven tradeoff reasoning) |
| 2.4 | Session save/resume | ✅ Implemented (CLI --resume, GUI save/load, full state serialization) |
| 2.5 | Report generation | ✅ Implemented |
| 2.6 | Stress test mode (D) | ✅ Implemented (adversarial prompt template, --search-mode stress_test) |
| 2.7 | Replay mode | Replay completed sessions in the GUI |

### Phase 3 — Additional ExaGO Applications

| Step | Task | Details | Status |
|------|------|---------|--------|
| 3.1 | DCOPFLOW support | DC approximation; fast screening proxy; reuses OPFLOW parser with DC-aware summary, prompt, and metric filtering | ✅ |
| 3.2 | SCOPFLOW support | Contingency file management; per-contingency results | |
| 3.3 | TCOPFLOW support | Time-series profiles; multi-period results | |
| 3.4 | SOPFLOW support | Stochastic scenarios; uncertainty reasoning | |

### Phase 4 — PFLOW-Based Direct Optimization

| Step | Task | Details |
|------|------|---------|
| 4.1 | PFLOW executor + parser | Power flow results extraction |
| 4.2 | Extended command set | Tap ratios, shunt switching, phase shifter control |
| 4.3 | Optimization prompt templates | Teach LLM basic heuristics: finite differences, bisection |
| 4.4 | Benchmark vs. OPFLOW | Compare LLM-driven PFLOW vs. OPFLOW optimal solutions |

### Phase 5 — ExaGO Integration

| Step | Task | Details |
|------|------|---------|
| 5.1 | Python API executor | ExaGO Python bindings; eliminate file I/O overhead |
| 5.2 | Large network support | Scalable metadata summarization; tool-use for 10k+ buses |
| 5.3 | Launcher integration | Add to Streamlit launcher |
| 5.4 | Visualization integration | Connect to Flask visualization system |

---

## 12. Integration with ExaGO Project

- **Shared file formats:** Same MATPOWER .m and .gic files.
- **ExaGO as external dependency:** Black-box binary during Phases 1–4.
- **Configuration portability:** Paths externalized to YAML.
- **Python API readiness:** Executor interface supports mode swap.
- **Conda compatibility:** Runs in exago312 environment (ExaGO) and llm-sim environment (LLM-Sim, Python 3.11).

---

## 13. Claude Code Workflow

Each implementation step follows:
1. **Architecture briefing:** This document provided as context.
2. **Task specification:** Specific prompt saved to `prompts/` folder with I/O expectations, interfaces, test criteria.
3. **Implementation:** Claude Code reads the prompt from file and writes code following project structure.
4. **Review and testing:** Verified by Claude (brainstorming instance) via Filesystem MCP inspection; issues result in follow-up prompts.
5. **Integration check:** New component verified against existing components.

This document serves as shared context between the human developer, Claude (brainstorming), Gemini (brainstorming), and Claude Code (implementation).

---

## 14. Key Lessons Learned

- **OPF voltage control:** In OPFLOW, `set_gen_voltage` only sets an initial guess — the solver overrides it. Voltage must be controlled via bus limits (`Vmin`/`Vmax`). This required adding explicit guidance in the system prompt, command schema warnings, and runtime feedback to the LLM.
- **Bulk commands matter:** Without `set_all_bus_vlimits`, the LLM could only constrain ~10 of 200 buses per iteration due to JSON output size limits. The bulk command was essential for the feasibility boundary search to succeed.
- **Violation reporting must use actual limits:** Hardcoded violation thresholds (0.9/1.1) masked constraint violations when the user or LLM set tighter limits. Using the actual bus `Vmin`/`Vmax` from the modified network fixed this.
- **Goal-type-aware reporting:** Always framing results as cost reduction is incorrect for feasibility boundary searches where cost increase is expected and desirable.
- **Model selection matters significantly:** GLM-5 achieves ~60% JSON reliability (wasting ~40% of iterations), while Claude Sonnet 4.6 offers near-perfect reliability at ~$0.70 per run. GLM-5.1 is currently incompatible.
- **LLM analysis vs deterministic code:** The LLM's free-text analysis often correctly identifies the best iteration even when the deterministic Python code selects the wrong one. The adopted principle: let the LLM determine what "best" means, with Python fallback if classification fails.
