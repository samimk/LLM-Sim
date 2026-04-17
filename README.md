# LLM-Sim

LLM-driven iterative simulation and analysis tool for the [ExaGO](https://github.com/ornl/ExaGO) power grid optimization toolkit.

LLM-Sim uses large language models to iteratively modify power grid simulation inputs, run ExaGO solvers, interpret results, and search for configurations that satisfy user-defined goals expressed in natural language.

## Quick Start

```bash
# Install
pip install -e .

# Run with a simple goal
llm-sim ./data/case_ACTIVSg200.m \
  "Find the maximum uniform load scaling factor before the system becomes infeasible"

# Dry run (validate config without executing)
llm-sim ./data/case_ACTIVSg200.m "test" --dry-run
```

## How It Works

LLM-Sim runs an iterative agent loop:

1. **Parse** the MATPOWER base case network (.m file)
2. **Run** a baseline simulation with ExaGO (OPFLOW, DCOPFLOW, or other supported application)
3. **Prompt** the LLM with the goal, network summary, and simulation results
4. **LLM decides** an action:
   - **modify** — apply network changes (load scaling, generator dispatch, branch status, etc.) and run a new simulation
   - **analyze** — request specific data (voltage profiles, line loading, etc.)
   - **complete** — report findings and terminate
5. **Repeat** steps 3-4 until the goal is achieved, determined infeasible, or max iterations reached

The search journal tracks every iteration, providing the LLM with a history of what has been tried and the results observed.

## Search Modes

- **Boundary finding** — "Find the maximum load scaling factor before infeasibility"
- **Scenario exploration** — "What happens if generator at bus 189 trips offline?"
- **Optimization** — "Minimize generation cost while keeping all voltages above 0.95 pu"
- **Multi-objective** — "Minimize cost while keeping voltages above 0.95 pu and line loadings below 85%"
- **Stress testing** — "Find the most critical N-1 contingencies by systematically testing line outages"
- **Analysis** — "Report the top 5 most congested transmission lines"

## Supported Applications

| Application | Description | Status |
|-------------|-------------|--------|
| **OPFLOW** | AC Optimal Power Flow — full nonlinear OPF with voltage magnitudes, reactive power, and cost optimization | ✅ Fully supported |
| **DCOPFLOW** | DC Optimal Power Flow — linearized approximation using phase angles and active power only. Faster than OPFLOW, useful for screening and contingency ranking | ✅ Fully supported |
| **SCOPFLOW** | Security-Constrained OPF — finds a preventive dispatch that survives all contingencies in a `.cont` file. Requires a contingency file | ✅ Fully supported |
| TCOPFLOW | Multi-Period OPF — time-coupled optimization with load profiles | Planned (Phase 3) |
| SOPFLOW | Stochastic OPF — optimization under uncertainty with scenario files | Planned (Phase 3) |
| PFLOW | Power Flow — no optimization, LLM performs the search directly | Planned (Phase 4) |

### DCOPFLOW vs OPFLOW

DCOPFLOW uses the DC power flow approximation:
- All bus voltages are fixed at 1.0 pu — voltage magnitude is not an optimization variable
- Reactive power (Q) is ignored — only active power (P) is optimized
- Simulations run significantly faster (typically 10-50x) than full AC OPF
- Voltage-related commands (`set_gen_voltage`, `set_bus_vlimits`, `set_all_bus_vlimits`) are automatically skipped with a warning
- Best suited for: fast screening, load scaling studies, contingency ranking, active power market analysis

Select the application via CLI (`--app dcopflow`) or in the launcher GUI dropdown.

### SCOPFLOW (Security-Constrained OPF)

SCOPFLOW optimizes the base case dispatch so that the network remains feasible even if any contingency in the contingency file occurs:
- Requires a `.cont` contingency file listing branch and generator outages
- The cost is typically higher than unconstrained OPFLOW — this "security premium" is the price of reliability
- Results show the **base case** operating point (the preventive dispatch), not individual contingency outcomes
- All OPFLOW commands work with SCOPFLOW (voltage control, load scaling, generator dispatch, etc.)
- Branch status commands (`set_branch_status`) permanently modify the topology — they do NOT simulate contingencies (the `.cont` file handles that)

Select via CLI (`--app scopflow --ctgc data/case_ACTIVSg200.cont`) or in the launcher GUI (application dropdown + contingency file selector).

## Multi-Objective Tracking

LLM-Sim can track multiple objectives simultaneously and reason about tradeoffs between them. Objectives can be introduced in three ways:

- **From the initial goal** — the LLM extracts objectives automatically (e.g., "minimize cost while keeping voltages above 0.95" registers cost as primary and voltage as a constraint)
- **Via steering** — inject a directive mid-search like "also track line loading" to add a secondary objective
- **LLM-proposed** — the agent itself can propose tracking a new metric when it notices a tension (e.g., cost decreasing but voltage stability degrading)

Tracked objectives are shown in a multi-objective trend chart in the GUI and included in PDF reports. The LLM receives a structured summary of how all tracked metrics evolve across iterations, enabling it to articulate tradeoffs and make informed decisions. At the end of a search, the post-search analysis identifies the key tradeoffs and can recommend multiple solutions representing different points on the tradeoff space.

The system includes 14 built-in metric extractors (generation cost, voltage deviation, line loading, active losses, generation reserve, and more). For simple single-objective goals, this infrastructure is transparent — everything works exactly as before.

## Stress Test Mode

LLM-Sim includes a dedicated stress test mode for adversarial contingency exploration. When activated, the LLM acts as a security analyst, systematically disabling network components to identify critical vulnerabilities.

```bash
# CLI
llm-sim ./data/case_ACTIVSg200.m \
  "Find the most critical N-1 contingencies" \
  --search-mode stress_test
```

In stress test mode, the LLM always uses fresh mode (each contingency tested independently from the base case), starts with the most loaded lines, and can escalate to N-2 combinations. The post-search report ranks contingencies by severity: infeasibility > voltage violations > high line loading > cost increase.

## Session Save/Resume

Searches can be saved to disk and resumed later — useful for long runs, interrupted sessions, or exploring different strategies from the same checkpoint.

### CLI

```bash
# During a running search, type 'save' in the terminal:
save
# Output: [Steering] Session saved to: workdir/saved_session_20260414_150000

# Resume later:
llm-sim --resume workdir/saved_session_20260414_150000 --config configs/local_config.yaml
```

When resuming, the goal and journal are loaded from the saved session, but the LLM backend and config settings come from the current config/CLI arguments — so you can resume with a different model or temperature.

### GUI

The launcher sidebar includes a "Session Save/Resume" section with a Save button (available after search completes or while running) and a dropdown to resume from previously saved sessions.

## Interactive Steering

While a search is running, you can inject steering directives from the terminal (CLI) or the GUI — without stopping and restarting the search.

### CLI steering commands

When running interactively (stdin is a TTY), a background listener accepts these commands:

| Input | Action |
|-------|--------|
| `<text>` | Inject an **augment** directive — the LLM considers it alongside the original goal |
| `replace: <text>` | Inject a **replace** directive — the LLM treats it as a new primary goal |
| `pause` | Pause the search at the next iteration boundary |
| `resume` | Resume a paused search |
| `stop` | Request graceful termination |
| `status` | Print current pause state and the last 3 injected directives |
| `save` | Save the current session state to disk for later resumption |

**Augment vs. replace semantics:**
- **Augment** — adds a constraint or preference to the current goal without discarding it. Example: `Focus on buses in area 3`.
- **Replace** — supersedes the current goal entirely. Example: `replace: Minimize voltage violations, ignore cost`.

Entering a directive while paused automatically resumes the search.

### GUI steering panel

The Streamlit launcher exposes the same capabilities via a steering panel in the live monitor. See [launcher/README.md](launcher/README.md) for details.

---

## Usage

### Shell script (recommended for interactive use)

`run_llm_sim.sh` is the easiest way to start a session. It prompts you to type
the goal interactively, prints a confirmation header, and then launches the
simulation.

```bash
# Use all defaults (configs/local_config.yaml, case_ACTIVSg200.m, 20 iterations)
./run_llm_sim.sh

# Override the config file only
./run_llm_sim.sh configs/my_config.yaml

# Override config and case file
./run_llm_sim.sh configs/my_config.yaml ./data/case_RTS.m

# Override all three (config, case file, max iterations)
./run_llm_sim.sh configs/my_config.yaml ./data/case_RTS.m 10
```

When run, the script will ask:
```
Enter simulation prompt: Find the maximum load scaling factor before infeasibility
```

Then print a summary before executing:
```
============================================================
  LLM-Sim Run
============================================================
  Config:    configs/local_config.yaml
  Case file: ./data/case_ACTIVSg200.m
  Max iter:  20
  Prompt:    Find the maximum load scaling factor before infeasibility
============================================================
```

The three positional arguments correspond to the three most commonly varied
settings. Everything else (backend, model, application, verbosity) is
controlled by the config file.

### Direct CLI

```bash
# Basic run
llm-sim ./data/case_ACTIVSg200.m "Find the maximum load scaling factor"

# With options
llm-sim ./data/case_ACTIVSg200.m "Minimize generation cost" \
  --backend anthropic --model claude-sonnet-4-20250514 \
  --app opflow --max-iter 30 --verbose

# Quiet mode (only show final summary)
llm-sim ./data/case_ACTIVSg200.m "Analyze voltage profile" --quiet

# Stress test mode (adversarial contingency exploration)
llm-sim ./data/case_ACTIVSg200.m \
  "Find critical N-1 contingencies" --search-mode stress_test

# Resume a saved session
llm-sim --resume workdir/saved_session_20260414_150000

# DC Optimal Power Flow (fast screening)
llm-sim ./data/case_ACTIVSg200.m \
  "Find the maximum load scaling factor before infeasibility" \
  --app dcopflow --max-iter 10 --mode fresh

# Security-Constrained OPF (requires contingency file)
llm-sim ./data/case_ACTIVSg200.m \
  "Find the minimum cost dispatch that survives all N-1 contingencies" \
  --app scopflow --ctgc data/case_ACTIVSg200.cont --max-iter 10

# Dry run (validate config without executing)
python -m llm_sim ./data/case_ACTIVSg200.m "test goal" --dry-run
```

## Example Output

```
============================================================
  LLM-Sim — LLM-driven iterative simulation for ExaGO
  Version 0.1.0
============================================================
  Backend:        anthropic
  Model:          claude-sonnet-4-20250514
  Application:    opflow
  Base case:      data/case_ACTIVSg200.m
  Goal:           Find the maximum load scaling factor
  Max iterations: 20
  Mode:           accumulative
============================================================

[Iter 0] Running base case simulation...
[Iter 0] Base case: CONVERGED, cost=$27,557.57

[Iter 1] Sending prompt to anthropic (claude-sonnet-4-20250514)...
[Iter 1] LLM action: modify — "Scale all loads +20%"
[Iter 1] Applied 1 command(s), 0 skipped
[Iter 1] Simulation completed in 0.04s — CONVERGED, cost=$33,019.55

...

[Iter 5] LLM action: complete
[Iter 5] Search completed: "Maximum feasible load increase is ~27%."

============================================================
  LLM-Sim Search Complete
============================================================
  Goal:           Find the maximum load scaling factor
  Application:    opflow
  Backend:        anthropic (claude-sonnet-4-20250514)
  Iterations:     6 (of max 20)
  Duration:       18.3 seconds
  Tokens used:    ~12,450 (prompt: 9,200, completion: 3,250)
  Termination:    completed
  Best objective: $27,557.57 (iteration 1)

  Findings: Maximum feasible uniform load increase is approximately 27%.
============================================================
```

## Installation

```bash
cd LLM-Sim
pip install -e .
```

Copy or symlink ExaGO binaries into `applications/` (see [applications/README.md](applications/README.md)) and place network data files in `data/` (see [data/README.md](data/README.md)).

## Configuration

Edit `configs/default_config.yaml` or pass `--config path/to/config.yaml`. CLI arguments override config file values.

Set your API key as an environment variable:
```bash
export ANTHROPIC_API_KEY="your-key-here"
# or
export OPENAI_API_KEY="your-key-here"
```

## Testing

```bash
# Run all unit tests
python -m pytest tests/ -v

# Run multi-objective tracking tests
python -m pytest tests/test_multi_objective.py -v

# Run session save/resume tests
python -m pytest tests/test_session_io.py -v

# Run DCOPFLOW-specific tests
python -m pytest tests/test_dcopflow.py -v

# Run SCOPFLOW-specific tests
python -m pytest tests/test_scopflow.py -v

# Run end-to-end tests (requires opflow binary)
python -m pytest tests/test_e2e.py -v -m "not slow"

# Run real LLM integration tests (requires API key + opflow)
python -m pytest tests/test_e2e.py -v
```

## Architecture

See [llm_sim_architecture.md](llm_sim_architecture.md) for the full design document.
