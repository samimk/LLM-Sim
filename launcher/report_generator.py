"""PDF report generator for completed LLM-Sim search sessions.

Uses ReportLab with DejaVu Sans font for diacritics support.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from llm_sim.engine.agent_loop import SearchSession
from llm_sim.parsers.opflow_results import OPFLOWResult

try:
    from charts import (
        convergence_chart, voltage_range_chart, voltage_profile_chart,
        generator_dispatch_chart, line_loading_chart,
    )
except ModuleNotFoundError:
    from launcher.charts import (
        convergence_chart, voltage_range_chart, voltage_profile_chart,
        generator_dispatch_chart, line_loading_chart,
    )

logger = logging.getLogger("launcher.report_generator")

# ── Font Registration ────────────────────────────────────────────────────────

_FONT_NAME = "DejaVuSans"
_FONT_REGISTERED = False


def _register_fonts():
    """Register DejaVu Sans font if available."""
    global _FONT_REGISTERED
    if _FONT_REGISTERED:
        return
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    bold_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]
    for p in font_paths:
        if Path(p).exists():
            pdfmetrics.registerFont(TTFont("DejaVuSans", p))
            for bp in bold_paths:
                if Path(bp).exists():
                    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", bp))
                    break
            _FONT_REGISTERED = True
            logger.info("Registered DejaVu Sans font from %s", p)
            return
    logger.warning("DejaVu Sans not found; using Helvetica fallback")


# ── Chart Export Helper ──────────────────────────────────────────────────────

def _export_chart_image(fig, width_px: int = 800, height_px: int = 400) -> bytes | None:
    """Export a Plotly figure to PNG bytes.

    Returns None if export fails (e.g., kaleido not installed).
    """
    try:
        return fig.to_image(format="png", width=width_px, height=height_px, scale=2)
    except Exception as exc:
        logger.warning("Failed to export chart image: %s", exc)
        return None


# ── Markdown Table Helpers ────────────────────────────────────────────────────

def _is_separator_row(line: str) -> bool:
    """Check if a line is a markdown table separator (e.g., |------|------|)."""
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    content = stripped.replace("|", "").replace(" ", "").replace("-", "").replace(":", "")
    return len(content) == 0 and "-" in stripped


def _parse_markdown_table(text: str) -> list[list[str]] | None:
    """Parse a markdown table from text into a list of rows.

    Returns None if the text is not a markdown table.
    The separator row (with dashes) is excluded.
    """
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return None

    # All lines must contain |
    if not all("|" in l for l in lines):
        return None

    rows = []
    for line in lines:
        if _is_separator_row(line):
            continue
        cells = [c.strip() for c in line.split("|")]
        # Strip empty cells from leading/trailing |
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if cells:
            rows.append(cells)

    return rows if len(rows) >= 2 else None


# ── Report Generator ─────────────────────────────────────────────────────────

class ReportGenerator:
    """Generates PDF reports from completed search sessions."""

    def __init__(self):
        _register_fonts()
        self._font = "DejaVuSans" if _FONT_REGISTERED else "Helvetica"
        self._font_bold = "DejaVuSans-Bold" if _FONT_REGISTERED else "Helvetica-Bold"
        self._styles = self._build_styles()

    @staticmethod
    def _escape_xml(text: str) -> str:
        """Escape XML special characters for ReportLab Paragraphs."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _is_ascii_art(self, text: str) -> bool:
        """Check if text contains ASCII art (box-drawing characters, etc.)."""
        art_chars = set("┤├┬┴┼─│┐┘┌└╭╮╯╰✗✓✕✔▉▊▋▌▍▎▏")
        char_count = sum(1 for c in text if c in art_chars)
        return char_count > 5

    def _preprocess_text(self, text: str) -> str:
        """Preprocess markdown text to normalize paragraph boundaries."""
        lines = text.split("\n")
        result = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                if result and result[-1].strip():
                    result.append("")
                result.append(line)
                result.append("")
            elif stripped == "---":
                if result and result[-1].strip():
                    result.append("")
                result.append("")
            else:
                result.append(line)
        return "\n".join(result)

    def _build_styles(self) -> dict[str, ParagraphStyle]:
        """Create custom paragraph styles."""
        return {
            "title": ParagraphStyle(
                "title", fontName=self._font_bold, fontSize=24,
                alignment=TA_CENTER, spaceAfter=12,
            ),
            "subtitle": ParagraphStyle(
                "subtitle", fontName=self._font, fontSize=14,
                alignment=TA_CENTER, spaceAfter=6, textColor=colors.grey,
            ),
            "heading1": ParagraphStyle(
                "heading1", fontName=self._font_bold, fontSize=18,
                spaceBefore=20, spaceAfter=10,
            ),
            "heading2": ParagraphStyle(
                "heading2", fontName=self._font_bold, fontSize=14,
                spaceBefore=14, spaceAfter=8,
            ),
            "body": ParagraphStyle(
                "body", fontName=self._font, fontSize=10,
                spaceAfter=6, leading=14,
            ),
            "body_small": ParagraphStyle(
                "body_small", fontName=self._font, fontSize=8,
                spaceAfter=4, leading=10,
            ),
            "caption": ParagraphStyle(
                "caption", fontName=self._font, fontSize=9,
                textColor=colors.grey, spaceAfter=4,
            ),
            "bullet": ParagraphStyle(
                "bullet", fontName=self._font, fontSize=10,
                spaceAfter=3, leading=14,
                leftIndent=15, bulletIndent=5,
                bulletFontName=self._font, bulletFontSize=10,
            ),
        }

    def generate(
        self,
        session: SearchSession,
        summary_text: str | None = None,
        base_result: OPFLOWResult | None = None,
        best_result: OPFLOWResult | None = None,
        goal_classification: dict | None = None,
    ) -> bytes:
        """Generate a PDF report and return it as bytes.

        Args:
            session: Completed search session.
            summary_text: Optional LLM-generated summary analysis.
            base_result: Base case OPFLOW results (for charts).
            best_result: Best feasible OPFLOW results (for charts).
            goal_classification: Optional dict with goal_type, best_iteration,
                best_iteration_rationale from LLM analysis.

        Returns:
            PDF file contents as bytes.
        """
        gc = goal_classification
        best_iter_override = gc.get("best_iteration") if gc else None
        goal_type = gc.get("goal_type") if gc else None

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            leftMargin=2 * cm, rightMargin=2 * cm,
            topMargin=2 * cm, bottomMargin=2 * cm,
        )

        story: list = []
        story.extend(self._build_title_page(session, goal_type=goal_type))
        story.append(PageBreak())
        story.extend(self._build_executive_summary(
            session, summary_text, goal_classification=gc,
        ))
        story.append(PageBreak())
        story.extend(self._build_convergence_section(
            session, best_iteration=best_iter_override,
        ))
        story.append(PageBreak())
        story.extend(self._build_comparison_section(
            session, base_result, best_result, goal_type=goal_type,
            best_iteration_override=best_iter_override,
        ))
        story.append(PageBreak())
        story.extend(self._build_iteration_log(session))

        doc.build(story)
        return buffer.getvalue()

    # ── Markdown / Summary Text Rendering ────────────────────────────────

    def _build_markdown_table(self, rows: list[list[str]]) -> Table:
        """Convert parsed markdown table rows into a styled ReportLab Table."""
        s = self._styles
        n_cols = max(len(r) for r in rows)
        padded = [r + [""] * (n_cols - len(r)) for r in rows]

        header_style = ParagraphStyle(
            "table_header", parent=s["body_small"],
            fontName=self._font_bold, fontSize=8,
            textColor=colors.white, leading=10,
        )
        cell_style = ParagraphStyle(
            "table_cell", parent=s["body_small"],
            fontName=self._font, fontSize=8, leading=10,
        )

        table_data = []
        for row_idx, row in enumerate(padded):
            style = header_style if row_idx == 0 else cell_style
            table_data.append([
                Paragraph(self._escape_xml(cell), style) for cell in row
            ])

        available_width = 17 * cm
        col_width = available_width / n_cols
        table = Table(table_data, colWidths=[col_width] * n_cols)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        return table

    def _render_summary_text(self, text: str) -> list:
        """Render summary text, converting markdown tables to ReportLab Tables."""
        text = self._preprocess_text(text)
        elements: list = []
        lines = text.split("\n")
        current_block: list[str] = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check for code block start
            if line.strip().startswith("```"):
                if current_block:
                    elements.extend(self._render_text_block("\n".join(current_block)))
                    current_block = []

                code_lines = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    i += 1  # Skip closing ```

                code_text = "\n".join(code_lines).strip()
                if code_text and not self._is_ascii_art(code_text):
                    escaped = self._escape_xml(code_text)
                    code_style = ParagraphStyle(
                        "code_block", fontName="Courier", fontSize=7,
                        spaceAfter=6, leading=9,
                        backColor=colors.HexColor("#f5f5f5"),
                        leftIndent=10, rightIndent=10,
                        spaceBefore=4,
                    )
                    elements.append(Paragraph(escaped.replace("\n", "<br/>"), code_style))
                continue

            # Check if this line starts a markdown table
            if "|" in line and i + 1 < len(lines) and _is_separator_row(lines[i + 1]):
                if current_block:
                    elements.extend(self._render_text_block("\n".join(current_block)))
                    current_block = []

                table_lines = [line]
                i += 1
                while i < len(lines) and "|" in lines[i]:
                    table_lines.append(lines[i])
                    i += 1

                rows = _parse_markdown_table("\n".join(table_lines))
                if rows:
                    elements.append(Spacer(1, 0.3 * cm))
                    elements.append(self._build_markdown_table(rows))
                    elements.append(Spacer(1, 0.3 * cm))
                continue

            current_block.append(line)
            i += 1

        if current_block:
            elements.extend(self._render_text_block("\n".join(current_block)))

        return elements

    def _render_text_block(self, text: str) -> list:
        """Render a non-table text block as Paragraph flowables."""
        s = self._styles
        elements: list = []

        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue

            # Skip ASCII art
            if self._is_ascii_art(para):
                continue

            # Headings
            if para.startswith("### "):
                heading_text = self._escape_xml(para[4:].strip())
                elements.append(Paragraph(heading_text, s["heading2"]))
            elif para.startswith("## "):
                heading_text = self._escape_xml(para[3:].strip())
                elements.append(Paragraph(heading_text, s["heading1"]))
            elif para.startswith("# "):
                heading_text = self._escape_xml(para[2:].strip())
                elements.append(Paragraph(heading_text, s["heading1"]))
            elif para.startswith("```"):
                # Code blocks that weren't caught at line level
                code_lines = para.split("\n")
                code_content = "\n".join(
                    l for l in code_lines if not l.strip().startswith("```")
                )
                if code_content.strip() and not self._is_ascii_art(code_content):
                    escaped = self._escape_xml(code_content)
                    code_style = ParagraphStyle(
                        "code", fontName="Courier", fontSize=8,
                        spaceAfter=6, leading=10,
                        backColor=colors.HexColor("#f5f5f5"),
                        leftIndent=10,
                    )
                    elements.append(Paragraph(escaped.replace("\n", "<br/>"), code_style))
            else:
                # Check for bullet lists
                lines = para.split("\n")
                bullet_lines = [l for l in lines if l.strip().startswith("- ")]
                if len(bullet_lines) > len(lines) / 2:
                    for line in lines:
                        line = line.strip()
                        if line.startswith("- "):
                            bullet_text = self._escape_xml(line[2:].strip())
                            elements.append(Paragraph(f"\u2022 {bullet_text}", s["bullet"]))
                        elif line:
                            elements.append(Paragraph(self._escape_xml(line), s["body"]))
                    continue

                # Regular paragraph
                cleaned = para.replace("**", "")
                cleaned = self._escape_xml(cleaned)
                cleaned = " ".join(cleaned.split("\n"))
                elements.append(Paragraph(cleaned, s["body"]))

        return elements

    # ── Title Page ───────────────────────────────────────────────────────

    def _build_title_page(
        self, session: SearchSession, goal_type: str | None = None,
    ) -> list:
        s = self._styles
        elements: list = []
        elements.append(Spacer(1, 6 * cm))
        elements.append(Paragraph("LLM-Sim Search Report", s["title"]))
        if goal_type:
            type_label = goal_type.replace("_", " ").title()
            elements.append(Paragraph(f"Search Type: {type_label}", s["caption"]))
        elements.append(Spacer(1, 1 * cm))
        elements.append(Paragraph(self._escape_xml(session.goal), s["subtitle"]))
        elements.append(Spacer(1, 2 * cm))

        start = datetime.fromisoformat(session.start_time)
        elements.append(Paragraph(
            f"Date: {start.strftime('%Y-%m-%d %H:%M:%S')}", s["body"],
        ))
        elements.append(Paragraph(
            f"Application: {session.application}", s["body"],
        ))
        elements.append(Paragraph(
            f"Backend: {session.config.llm.backend} / {session.config.llm.model}", s["body"],
        ))
        elements.append(Spacer(1, 3 * cm))
        elements.append(Paragraph("Generated by LLM-Sim v0.1.0", s["caption"]))
        return elements

    # ── Executive Summary ────────────────────────────────────────────────

    def _build_executive_summary(
        self,
        session: SearchSession,
        summary_text: str | None,
        goal_classification: dict | None = None,
    ) -> list:
        s = self._styles
        elements: list = []
        elements.append(Paragraph("Executive Summary", s["heading1"]))

        gc = goal_classification
        best_iter_override = gc.get("best_iteration") if gc else None
        goal_type = gc.get("goal_type") if gc else None

        stats = session.journal.summary_stats(
            best_iteration_override=best_iter_override,
            goal_type=goal_type,
        )
        start = datetime.fromisoformat(session.start_time)
        end = datetime.fromisoformat(session.end_time) if session.end_time else datetime.now()
        duration = end - start
        total_tokens = session.total_prompt_tokens + session.total_completion_tokens

        # Key results
        lines = [
            f"Total iterations: {stats['total_iterations']}",
            f"Feasible solutions: {stats['feasible_count']}",
            f"Infeasible: {stats['infeasible_count']}",
            f"Duration: {duration.total_seconds():.0f}s",
            f"Termination: {session.termination_reason}",
            f"Token usage: {total_tokens:,}" if total_tokens > 0 else "Token usage: N/A",
        ]

        if stats["best_objective"] is not None:
            base_entry = session.journal.entries[0] if session.journal.entries else None
            best_str = f"Best objective: ${stats['best_objective']:,.2f} (iteration {stats['best_iteration']})"
            if base_entry and base_entry.objective_value and base_entry.objective_value != 0:
                pct = (base_entry.objective_value - stats["best_objective"]) / base_entry.objective_value * 100
                best_str += f" — {pct:.1f}% improvement vs base case"
            lines.insert(0, best_str)
        else:
            lines.insert(0, "No feasible solution found.")

        for line in lines:
            elements.append(Paragraph(self._escape_xml(line), s["body"]))

        # Goal achievement rationale
        if gc and gc.get("best_iteration_rationale"):
            rationale = self._escape_xml(gc["best_iteration_rationale"])
            elements.append(Paragraph(f"Goal achievement: {rationale}", s["body"]))

        # LLM Analysis
        if summary_text:
            elements.append(Spacer(1, 0.5 * cm))
            elements.append(Paragraph("Analysis", s["heading2"]))
            elements.extend(self._render_summary_text(summary_text))

        return elements

    # ── Convergence Section ──────────────────────────────────────────────

    def _build_convergence_section(
        self, session: SearchSession, best_iteration: int | None = None,
    ) -> list:
        s = self._styles
        elements: list = []
        elements.append(Paragraph("Convergence Analysis", s["heading1"]))

        stats = session.journal.summary_stats(best_iteration_override=best_iteration)

        # Auto-generated text
        n = stats["total_iterations"]
        text = f"The search ran for {n} iterations."
        if stats["best_objective"] is not None:
            base_entry = session.journal.entries[0] if session.journal.entries else None
            if base_entry and base_entry.objective_value and base_entry.objective_value != 0:
                pct = (base_entry.objective_value - stats["best_objective"]) / base_entry.objective_value * 100
                text += f" A {pct:.1f}% cost change was observed from the base case."
            text += f" The best solution (objective ${stats['best_objective']:,.2f}) was found at iteration {stats['best_iteration']}."
        elements.append(Paragraph(text, s["body"]))
        elements.append(Spacer(1, 0.5 * cm))

        # Convergence chart
        fig = convergence_chart(
            session.journal, highlight_best=True, height=350,
            best_iteration=best_iteration,
        )
        img_bytes = _export_chart_image(fig, width_px=700, height_px=350)
        if img_bytes:
            elements.append(Image(io.BytesIO(img_bytes), width=16 * cm, height=8 * cm))
            elements.append(Paragraph("Objective value convergence across iterations.", s["caption"]))
        elements.append(Spacer(1, 0.5 * cm))

        # Voltage range chart
        fig_v = voltage_range_chart(session.journal, height=300)
        img_bytes_v = _export_chart_image(fig_v, width_px=700, height_px=300)
        if img_bytes_v:
            elements.append(Image(io.BytesIO(img_bytes_v), width=16 * cm, height=7 * cm))
            elements.append(Paragraph("Voltage range envelope across iterations.", s["caption"]))

        return elements

    # ── Comparison Section ───────────────────────────────────────────────

    def _build_comparison_section(
        self,
        session: SearchSession,
        base_result: OPFLOWResult | None,
        best_result: OPFLOWResult | None,
        goal_type: str | None = None,
        best_iteration_override: int | None = None,
    ) -> list:
        s = self._styles
        elements: list = []

        heading = "Results Comparison"
        if goal_type == "feasibility_boundary":
            heading = "Base Case vs Maximum Feasible Configuration"
        elif goal_type == "constraint_satisfaction":
            heading = "Base Case vs Best Constraint-Satisfying Configuration"
        elif goal_type == "parameter_exploration":
            heading = "Base Case vs Selected Exploration Result"
        elements.append(Paragraph(heading, s["heading1"]))

        # Build comparison table from journal entries
        stats = session.journal.summary_stats(
            best_iteration_override=best_iteration_override,
            goal_type=goal_type,
        )
        base_entry = session.journal.entries[0] if session.journal.entries else None
        best_entry = None
        if stats.get("best_iteration") is not None:
            for e in session.journal.entries:
                if e.iteration == stats["best_iteration"]:
                    best_entry = e
                    break

        def _fv(v, fmt=".2f"):
            return f"{v:{fmt}}" if v is not None and v != 0 else "—"

        header = ["Metric", "Base Case", "Best Solution", "Change"]
        rows = [header]

        if base_entry:
            bv = base_entry.objective_value
            sv = best_entry.objective_value if best_entry else None
            change = ""
            if bv is not None and sv is not None and bv != 0:
                pct = (sv - bv) / bv * 100
                change = f"{pct:+.1f}%"
            rows.append([
                "Objective ($)",
                f"${bv:,.2f}" if bv is not None else "—",
                f"${sv:,.2f}" if sv is not None else "N/A",
                change or "—",
            ])
            rows.append([
                "Generation (MW)",
                _fv(base_entry.total_gen_mw, ".1f"),
                _fv(best_entry.total_gen_mw, ".1f") if best_entry else "N/A",
                f"{best_entry.total_gen_mw - base_entry.total_gen_mw:+.1f}" if best_entry and base_entry.total_gen_mw else "—",
            ])
            rows.append([
                "Voltage Min (p.u.)",
                _fv(base_entry.voltage_min, ".4f"),
                _fv(best_entry.voltage_min, ".4f") if best_entry else "N/A",
                "—",
            ])
            rows.append([
                "Voltage Max (p.u.)",
                _fv(base_entry.voltage_max, ".4f"),
                _fv(best_entry.voltage_max, ".4f") if best_entry else "N/A",
                "—",
            ])
            rows.append([
                "Max Line Loading (%)",
                _fv(base_entry.max_line_loading_pct, ".1f"),
                _fv(best_entry.max_line_loading_pct, ".1f") if best_entry else "N/A",
                "—",
            ])
            rows.append([
                "Violations",
                str(base_entry.violations_count),
                str(best_entry.violations_count) if best_entry else "N/A",
                str(best_entry.violations_count - base_entry.violations_count) if best_entry else "—",
            ])

        table = Table(rows, colWidths=[5 * cm, 4 * cm, 4 * cm, 3 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), self._font_bold),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 1), (-1, -1), self._font),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 1 * cm))

        # Voltage profile chart
        fig_vp = voltage_profile_chart(base_result, best_result)
        if fig_vp is not None:
            img_bytes = _export_chart_image(fig_vp, width_px=700, height_px=400)
            if img_bytes:
                elements.append(Image(io.BytesIO(img_bytes), width=16 * cm, height=9 * cm))
                elements.append(Paragraph("Bus voltage profile comparison.", s["caption"]))
                elements.append(Spacer(1, 0.5 * cm))

        # Generator dispatch chart
        fig_gen = generator_dispatch_chart(base_result, best_result)
        if fig_gen is not None:
            img_bytes = _export_chart_image(fig_gen, width_px=700, height_px=400)
            if img_bytes:
                elements.append(Image(io.BytesIO(img_bytes), width=16 * cm, height=9 * cm))
                elements.append(Paragraph("Generator dispatch comparison.", s["caption"]))

        return elements

    # ── Iteration Log ────────────────────────────────────────────────────

    def _build_iteration_log(self, session: SearchSession) -> list:
        s = self._styles
        elements: list = []
        elements.append(Paragraph("Iteration Log", s["heading1"]))

        header = ["Iter", "Description", "Cost ($)", "Feas.", "V_min", "V_max", "Load%", "Time(s)"]
        rows = [header]

        for e in session.journal.entries:
            rows.append([
                str(e.iteration),
                e.description[:40],
                f"${e.objective_value:,.2f}" if e.objective_value is not None else "FAILED",
                "Y" if e.feasible else "N",
                f"{e.voltage_min:.3f}" if e.voltage_min > 0 else "—",
                f"{e.voltage_max:.3f}" if e.voltage_max > 0 else "—",
                f"{e.max_line_loading_pct:.1f}" if e.max_line_loading_pct > 0 else "—",
                f"{e.elapsed_seconds:.1f}",
            ])

        col_widths = [1.2 * cm, 5.5 * cm, 2.8 * cm, 1.2 * cm, 1.8 * cm, 1.8 * cm, 1.5 * cm, 1.5 * cm]
        table = Table(rows, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), self._font_bold),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 1), (-1, -1), self._font),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elements.append(table)

        return elements
