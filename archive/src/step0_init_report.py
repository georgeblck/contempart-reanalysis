"""Initialize a fresh report. Run this first to clear any previous results."""

from .report import Report

report = Report()
report.init_report()
print("Ready. Run steps 1-4 in order.")
