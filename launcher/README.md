# LLM-Sim Launcher

A Streamlit-based GUI for configuring, running, and monitoring LLM-Sim search sessions. The launcher provides a web interface for setting up simulations, watching live progress, exploring results with interactive charts, and generating PDF reports.

## Prerequisites

- Python 3.10+
- LLM-Sim installed from the project root: `pip install -e .`
- ExaGO binaries configured in `applications/` (see main project README)
- An LLM API key (Anthropic or OpenAI) set as an environment variable

## Installation

```bash
pip install -r launcher/requirements.txt
```

This installs Streamlit, Plotly, kaleido (for PDF chart export), ReportLab (for PDF generation), and PyYAML.

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

### Configuration Panel (Sidebar)
- Select MATPOWER base case files from the `data/` directory
- Choose LLM backend (Anthropic, OpenAI, Ollama, Ollama-Cloud) with auto-populated model defaults
- Adjust temperature, search mode (accumulative/fresh), and max iterations
- Preset goal library with common optimization tasks (minimize cost, fix voltage violations, etc.)
- Custom goal input via free-text area

### Live Search Monitor
- Two-column layout: iteration timeline (left) and live charts (right)
- Expandable iteration cards showing LLM reasoning, commands, and key metrics
- Real-time convergence chart (objective value vs iteration, color-coded by feasibility)
- Live voltage range chart with limit reference lines
- Progress stats: iteration count, feasible count, best cost found
- Phase status indicator (sending prompt, running simulation, parsing results, etc.)
- Stop button for graceful search termination

### Results & Summary View (Three Tabs)
- **Overview**: Summary metrics, base-vs-best comparison table, convergence chart, voltage range chart
- **Detailed Results**: Voltage profile comparison, generator dispatch chart, line loading chart, full iteration history table
- **Analysis & Report**: On-demand LLM-generated analytical summary, auto-generated search narrative, PDF report download

### PDF Reports
- Professional multi-page PDF with title page, executive summary, convergence charts, results comparison tables, and full iteration log
- Uses DejaVu Sans font for diacritics support
- Chart images exported via Plotly/kaleido

### Session History
- Completed sessions are tracked in the sidebar for reference during a browser session

## Configuration

The GUI widget values override defaults from `configs/default_config.yaml`. The override mechanism uses dot-notation keys (e.g., `llm.backend`, `search.max_iterations`) passed to `llm_sim.config.load_config(cli_overrides=...)`.

Key configuration paths:
- **Base config**: `configs/default_config.yaml`
- **Data files**: `data/*.m` (MATPOWER format)
- **Applications**: `applications/` (ExaGO binaries)
- **Working directory**: `workdir/` (created at runtime)

## Troubleshooting

### "No .m files found in data/ directory"
Ensure MATPOWER `.m` files are present in the `data/` directory at the project root.

### "ANTHROPIC_API_KEY not set" / "OPENAI_API_KEY not set"
Export the relevant API key before launching:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```
Or configure it in your `env_setup.sh` script.

### "DejaVu Sans not found" (PDF generation warning)
The PDF generator uses DejaVu Sans for diacritics support. Install the font package:
- **openSUSE**: `sudo zypper install dejavu-sans-fonts`
- **Debian/Ubuntu**: `sudo apt install fonts-dejavu-core`
- **Fedora**: `sudo dnf install dejavu-sans-fonts`

The generator falls back to Helvetica if DejaVu Sans is not found.

### "Failed to export chart image" (PDF charts missing)
Install kaleido for Plotly image export:
```bash
pip install kaleido
```

### Config paths not resolving / import errors
Ensure you run the launcher from the **project root**, not from inside `launcher/`. Use `./launcher/run.sh` which handles this automatically.

### Streamlit port conflict
If port 8501 is in use, specify an alternative:
```bash
streamlit run launcher/app.py --server.port 8502
```
