# LLM-Sim — Architecture Document

**LLM-Driven Iterative Simulation and Analysis for ExaGO**

| Field | Value |
|-------|-------|
| Project | LLM-Sim (to be integrated into ExaGO) |
| Version | 0.1 — Proof of Concept |
| Date | March 2026 |
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
- **Designed for future interactive steering:** The PoC runs autonomously once started, but the architecture anticipates mid-search user intervention.
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

1. Agent Loop Controller assembles prompt (goal + search journal + latest results)
2. LLM backend generates response with structured action (modify / complete / analyze)
3. Controller parses response; if action is "modify", extracts JSON commands
4. Modification Engine applies commands to a working copy of the base case
5. Simulation Executor invokes ExaGO with the modified input files
6. Results Parser extracts structured data from ExaGO output
7. Controller updates the Search Journal with iteration summary
8. Loop returns to step 1 (or terminates if action is "complete" or max iterations reached)

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
| Anthropic | anthropic | Claude Sonnet/Opus | JSON via system prompt |
| Ollama (local) | ollama | Qwen 2.5 7B/14B, Llama 3.x | JSON mode varies |
| Ollama Cloud | ollama-cloud | Same as local | Same as local |

Configuration via YAML file or environment variables. Temperature kept low (0.2–0.4).

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
| DCOPFLOW | dcopflow | .m | DC angles, flows, cost |
| SCOPFLOW | scopflow | .m + contingency | Base case + contingency results |
| TCOPFLOW | tcopflow | .m + load profile | Multi-period dispatch |
| SOPFLOW | sopflow | .m + scenarios | Stochastic results across scenarios |
| PFLOW | pflow | .m | Power flow solution (no optimization) |

### 4.3 Results Parser

Produces two outputs:
- **Full structured result:** Python dictionary with all extracted data. Stored for detailed queries.
- **Compact summary:** 15–30 lines of text for LLM prompt inclusion. Contains: objective value, feasibility status, violation count/severity, voltage range, top-5 loaded lines, total generation vs. load.

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

### 4.5 Agent Loop Controller

Responsibilities:
- **Prompt assembly:** Combines system prompt + goal + journal + latest results
- **Response parsing:** Extracts JSON from LLM output (handles markdown fences, extra text)
- **Action dispatch:** Routes "modify" → Modification Engine + Executor; "analyze" → detailed query; "complete" → terminate with report
- **Termination logic:** Stops on LLM completion, max iterations, or critical error
- **Modification mode management:** Supports "accumulative" and "fresh" modes

### 4.6 Modification Engine

Deterministic Python module that applies JSON commands to MATPOWER .m files:

| Command | Parameters | Description |
|---------|-----------|-------------|
| set_load | {bus, Pd, Qd} | Set load at a specific bus |
| scale_load | {area\|zone\|bus, factor} | Scale load by factor |
| scale_all_loads | {factor} | Scale all loads uniformly |
| set_gen_status | {bus, id?, status} | Turn generator on/off |
| set_gen_dispatch | {bus, id?, Pg} | Set generator power output |
| set_gen_voltage | {bus, id?, Vg} | Set generator voltage setpoint |
| set_branch_status | {fbus, tbus, ckt?, status} | Enable/disable branch |
| set_branch_rate | {fbus, tbus, ckt?, rateA} | Modify thermal rating |
| set_cost_coeffs | {bus, id?, coeffs} | Modify cost curve |
| set_bus_vlimits | {bus, Vmin, Vmax} | Set voltage bounds |
| set_ground_resistance | {substation, R_ground} | Modify grounding (GIC) |

All commands validated before application. Invalid commands reported to LLM as errors.

### 4.7 User Interface (CLI)

User provides: base case path, application (default: opflow), search goal (natural language), optional config (LLM backend, max iterations, mode). Progress displayed live; summary report on completion.

---

## 5. Data Flow and Iteration Lifecycle

1. **Prompt Construction** — Controller reads journal, formats compact table, combines with system prompt (command schema + network metadata), goal statement, and previous results summary.
2. **LLM Inference** — Prompt sent to selected backend. Expected response: JSON with "action" (modify|complete|analyze), "reasoning", and action-specific payload.
3. **Response Parsing** — JSON extracted, handling markdown fences and extra text. Parse failure → error logged, loop continues.
4. **Command Validation** — Each command validated against network data. Invalid → error message fed back to LLM.
5. **File Modification** — Valid commands applied to working copy. Written to timestamped directory.
6. **Simulation Execution** — ExaGO invoked. Timeout guard (default 120s).
7. **Results Parsing** — Full structured result + compact summary produced.
8. **Journal Update** — New entry appended with iteration metadata and results.
9. **Loop Decision** — Continue unless LLM declared complete or max iterations reached.

---

## 6. LLM Prompt Architecture

Four sections assembled per iteration:

**Section A — System Prompt (static):** Role definition, command schema with parameter types and validation rules, network metadata (bus count, generators with fuel types and capacities, areas, voltage levels), response format specification.

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

## 8. Interactive Steering (Future)

Capabilities planned:
- **Goal refinement:** "Focus more on voltage issues, cost is secondary."
- **Constraint injection:** "Don't touch generators in area 3."
- **Strategy override:** "Try reducing load instead of adding generation."
- **Detail requests:** "Show me the voltage profile for all 230kV buses."
- **Branching:** "Save this state and explore an alternative from iteration 5."

Implementation: non-blocking user input check between iterations; input prepended to next prompt as "operator directive."

---

## 9. Multi-Objective Decision Making (Future)

- **Tradeoff articulation:** LLM explains cost of moving between solutions in natural language.
- **Pareto approximation:** Systematic exploration presenting non-dominated solutions.
- **Preference elicitation:** Clarifying questions about user priorities via interactive steering.
- **Constraint softening:** Distinguishing hard constraints from soft preferences.

---

## 10. Phased Implementation Plan

### Phase 1 — Foundation + OPFLOW PoC

| Step | Task | Details | Category |
|------|------|---------|----------|
| 1.1 | Project scaffolding | Directory structure, config (YAML), CLI args, logging | Foundation |
| 1.2 | LLM backend abstraction | Base class + OpenAI, Anthropic, Ollama adapters | Core |
| 1.3 | MATPOWER file parser/writer | Read .m → Python structures; write back preserving format | Core |
| 1.4 | Modification engine | Apply JSON commands; validate; report errors | Core |
| 1.5 | Simulation executor (CLI) | Invoke opflow binary, capture output, handle timeouts | Core |
| 1.6 | OPFLOW results parser | Extract objective, voltages, dispatch, flows, convergence | Core |
| 1.7 | Search journal | In-memory structure with append, format, export | Core |
| 1.8 | Agent loop controller | Prompt assembly, response parsing, action dispatch, termination | Core |
| 1.9 | System prompt templates | OPFLOW prompts with command schema, network metadata | Prompts |
| 1.10 | CLI interface | Goal input, live progress, final report | Interface |
| 1.11 | End-to-end testing | ACTIVSg200 case, all backends, scenario + boundary goals | Validation |

### Phase 2 — Enrichment + Multi-Objective

| Step | Task | Details |
|------|------|---------|
| 2.1 | Interactive steering | Non-blocking input, operator directives |
| 2.2 | Analyze action | Detailed views: voltage profiles, gen summary, line loading |
| 2.3 | Multi-objective tracking | Journal extensions, Pareto front, preference prompts |
| 2.4 | Session save/resume | Serialize state to disk; resume from checkpoint |
| 2.5 | Report generation | PDF/Markdown reports with search trajectory charts |
| 2.6 | Stress test mode (D) | Adversarial prompts, combinatorial contingency exploration |

### Phase 3 — Additional ExaGO Applications

| Step | Task | Details |
|------|------|---------|
| 3.1 | DCOPFLOW support | DC approximation; fast screening proxy |
| 3.2 | SCOPFLOW support | Contingency file management; per-contingency results |
| 3.3 | TCOPFLOW support | Time-series profiles; multi-period results |
| 3.4 | SOPFLOW support | Stochastic scenarios; uncertainty reasoning |

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

## 11. Integration with ExaGO Project

- **Shared file formats:** Same MATPOWER .m and .gic files.
- **ExaGO as external dependency:** Black-box binary during Phases 1–4.
- **Configuration portability:** Paths externalized to YAML.
- **Python API readiness:** Executor interface supports mode swap.
- **Conda compatibility:** Runs in exago312 environment.

---

## 12. Claude Code Workflow

Each implementation step follows:
1. **Architecture briefing:** This document provided as context.
2. **Task specification:** Specific prompt with I/O expectations, interfaces, test criteria.
3. **Implementation:** Claude Code writes code following project structure.
4. **Review and testing:** Verified against criteria; issues fixed via follow-up.
5. **Integration check:** New component verified against existing components.

This document serves as shared context between the human developer, Claude (brainstorming), and Claude Code (implementation).
