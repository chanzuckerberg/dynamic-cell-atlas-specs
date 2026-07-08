#!/usr/bin/env python3
"""
Simple test runner for Dynamic Cell Atlas Specifications.

This script provides convenient commands to run different types of tests.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle the result."""
    print(f"\n🔍 {description}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"✅ {description} - PASSED")
    else:
        print(f"❌ {description} - FAILED")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Test runner for documentation")
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "all", "slow"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Ensure we're in the right directory
    if not Path("docs").exists() or not Path("tests").exists():
        print("❌ Error: Run this script from the repository root directory")
        sys.exit(1)

    # Check if dependencies are installed
    try:
        import pytest
        import yaml
    except ImportError:
        print("❌ Error: Missing dependencies. Install with:")
        print("   pip install -r docs/requirements.txt")
        sys.exit(1)

    success = True

    if args.type in ["unit", "all"]:
        cmd = ["pytest", "tests/", "-m", "unit"]
        if args.verbose:
            cmd.append("-v")
        success &= run_command(cmd, "Unit Tests")

    if args.type in ["integration", "all"]:
        cmd = ["pytest", "tests/", "-m", "integration"]
        if args.verbose:
            cmd.append("-v")
        success &= run_command(cmd, "Integration Tests")

    if args.type == "slow":
        cmd = ["pytest", "tests/", "-m", "slow"]
        if args.verbose:
            cmd.append("-v")
        success &= run_command(cmd, "Slow Tests")

    # Build documentation as a final check
    if args.type in ["integration", "all"]:
        cmd = ["sphinx-build", "-W", "-b", "html", "docs", "docs/_build/html"]
        success &= run_command(cmd, "Documentation Build")

    if success:
        print(f"\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
