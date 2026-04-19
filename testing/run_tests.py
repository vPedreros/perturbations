#!/usr/bin/env python3
"""Run all tests and print a summary report."""
import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

result = subprocess.run(
    [sys.executable, '-m', 'pytest', '.', '-v', '--tb=short', '--no-header'],
    capture_output=False
)
sys.exit(result.returncode)
