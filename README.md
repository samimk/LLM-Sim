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
2. **Run** a baseline simulation with ExaGO (e.g., OPFLOW)
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
- **Analysis** — "Report the top 5 most congested transmission lines"

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

# Run end-to-end tests (requires opflow binary)
python -m pytest tests/test_e2e.py -v -m "not slow"

# Run real LLM integration tests (requires API key + opflow)
python -m pytest tests/test_e2e.py -v
```

## Architecture

See [llm_sim_architecture.md](llm_sim_architecture.md) for the full design document.
