# LLM-Sim

LLM-driven iterative simulation and analysis tool for the [ExaGO](https://github.com/ornl/ExaGO) power grid optimization toolkit.

LLM-Sim uses large language models to iteratively modify power grid simulation inputs, run ExaGO solvers, interpret results, and search for configurations that satisfy user-defined goals expressed in natural language.

## Installation

```bash
# Clone and install
cd LLM-Sim
pip install -e .
```

Copy or symlink ExaGO binaries into `applications/` (see [applications/README.md](applications/README.md)) and place network data files in `data/` (see [data/README.md](data/README.md)).

## Usage

```bash
# Basic run
llm-sim ./data/case_ACTIVSg200.m "Find the maximum load scaling factor before infeasibility"

# With options
llm-sim ./data/case_ACTIVSg200.m "Minimize generation cost" \
  --backend anthropic --model claude-sonnet-4-20250514 \
  --app opflow --max-iter 30 --verbose

# Dry run (validate config without executing)
llm-sim ./data/case_ACTIVSg200.m "test goal" --dry-run --verbose

# Run as module
python -m llm_sim ./data/case_ACTIVSg200.m "test goal" --dry-run
```

## Configuration

Edit `configs/default_config.yaml` or pass `--config path/to/config.yaml`. CLI arguments override config file values.

## Architecture

See [llm_sim_architecture.md](llm_sim_architecture.md) for the full design document.
