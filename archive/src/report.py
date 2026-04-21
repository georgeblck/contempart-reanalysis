"""
Shared report utilities. Each pipeline step appends its section.

Usage from other scripts:
    from report import Report
    report = Report()
    report.header("## My Section")
    report.line("Some finding")
    report.table(["col1", "col2"], [["a", "b"], ["c", "d"]])
    report.image("plots/my_plot.png", "Caption")
    report.save()
"""

from datetime import datetime
from pathlib import Path


REPORT_PATH = Path("results/report.md")


class Report:
    def __init__(self, path: Path = REPORT_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lines: list[str] = []

    def header(self, text: str):
        self.lines.append(f"\n{text}\n")

    def line(self, text: str):
        self.lines.append(text)

    def blank(self):
        self.lines.append("")

    def table(self, headers: list[str], rows: list[list[str]]):
        self.lines.append("| " + " | ".join(headers) + " |")
        self.lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            self.lines.append("| " + " | ".join(str(x) for x in row) + " |")
        self.blank()

    def image(self, path: str, caption: str = ""):
        self.lines.append(f"![{caption}]({path})")
        self.blank()

    def save(self):
        """Append this section to the report file."""
        with open(self.path, "a") as f:
            f.write("\n".join(self.lines))
            f.write("\n")
        print(f"  -> Appended to {self.path}")

    def init_report(self):
        """Start a fresh report (call only once, before first step)."""
        with open(self.path, "w") as f:
            f.write(f"# contempArt CLIP Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write("Each section below was produced by a separate pipeline step.\n")
            f.write("Re-run any step to update its section (append mode).\n")
        print(f"Initialized {self.path}")
