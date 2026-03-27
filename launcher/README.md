# LLM-Sim Launcher

A Streamlit-based GUI for configuring, running, and monitoring LLM-Sim search sessions. The launcher provides a web interface for setting up simulations, watching live progress, exploring results with interactive charts, and generating PDF reports.

## Prerequisites

- Python 3.12+
- LLM-Sim installed from the project root: `pip install -e .`
- ExaGO binaries configured in `applications/` (see main project README)
- An LLM API key (Anthropic or OpenAI) set as an environment variable

## Installation

```bash
pip install -r launcher/requirements.txt
```

## Running

The launcher must be run from the **project root** directory so that config paths (`./applications`, `./data`, `./workdir`) resolve correctly.

```bash
# Recommended: use the launch script
./launcher/run.sh

# Or manually from project root
cd /path/to/LLM-Sim
streamlit run launcher/app.py
```

Do **not** run `streamlit run app.py` from inside `launcher/` — paths will not resolve.

## Features

- Configure searches via GUI: select base case, backend, model, goal, and search parameters
- Preset goal library with common analysis tasks
- Live monitoring of search iterations with progress updates
- Interactive results visualization (cost trend, voltage profiles, line loading)
- PDF report generation for completed sessions
