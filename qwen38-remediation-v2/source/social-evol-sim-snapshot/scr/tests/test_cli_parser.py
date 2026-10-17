"""Tests for CLI subcommand parsing."""

import sys
import subprocess
import pytest
from unittest.mock import patch
from scr.simulation.cli.cli_parser import parse_cli_args


class TestRunSubcommand:
    def test_run_basic(self):
        with patch.object(sys, "argv", ["main.py", "run", "--config_dir", "configZ_major_v2"]):
            ops, overrides = parse_cli_args()
        assert ops["mode"] == "run"
        assert ops["config_dir"] == "configZ_major_v2"
        assert overrides == {}

    def test_run_with_config_override(self):
        with patch.object(sys, "argv", [
            "main.py", "run", "--config_dir", "configZ_major_v2",
            "--config.world.max_life_steps", "5",
        ]):
            ops, overrides = parse_cli_args()
        assert ops["mode"] == "run"
        assert overrides["world.max_life_steps"] == 5

    def test_run_with_shared_flags(self):
        with patch.object(sys, "argv", [
            "main.py", "run", "--config_dir", "configZ_major_v2",
            "--dashboard", "--no_db", "--log_level", "debug",
        ]):
            ops, _ = parse_cli_args()
        assert ops["dashboard"] is True
        assert ops["no_db"] is True
        assert ops["log_level"] == "debug"

    def test_run_missing_config_dir_errors(self):
        with patch.object(sys, "argv", ["main.py", "run"]):
            with pytest.raises(SystemExit):
                parse_cli_args()

    def test_run_invalid_config_dir_errors(self):
        with patch.object(sys, "argv", ["main.py", "run", "--config_dir", "nonexistent_config_xyz"]):
            with pytest.raises(SystemExit):
                parse_cli_args()


class TestResumeSubcommand:
    def test_resume_basic(self):
        with patch.object(sys, "argv", ["main.py", "resume", "0307-121321"]):
            ops, overrides = parse_cli_args()
        assert ops["mode"] == "resume"
        assert ops["run_id"] == "0307-121321"
        assert overrides == {}

    def test_resume_with_time_step(self):
        with patch.object(sys, "argv", ["main.py", "resume", "0307-121321", "--time_step", "10"]):
            ops, _ = parse_cli_args()
        assert ops["run_id"] == "0307-121321"
        assert ops["time_step"] == 10

    def test_resume_with_config_override(self):
        with patch.object(sys, "argv", [
            "main.py", "resume", "0307-121321",
            "--config.llm.provider", "openrouter",
        ]):
            ops, overrides = parse_cli_args()
        assert ops["mode"] == "resume"
        assert overrides["llm.provider"] == "openrouter"

    def test_resume_does_not_require_config_dir(self):
        with patch.object(sys, "argv", ["main.py", "resume", "0307-121321"]):
            ops, _ = parse_cli_args()
        assert ops["config_dir"] is None


class TestListRunsSubcommand:
    def test_list_runs(self):
        with patch.object(sys, "argv", ["main.py", "list-runs"]):
            ops, overrides = parse_cli_args()
        assert ops["mode"] == "list-runs"
        assert overrides == {}


class TestEstimateCostSubcommand:
    def test_estimate_cost_basic(self):
        with patch.object(sys, "argv", ["main.py", "estimate-cost", "--config_dir", "configZ_major_v2"]):
            ops, _ = parse_cli_args()
        assert ops["mode"] == "estimate-cost"
        assert ops["config_dir"] == "configZ_major_v2"

    def test_estimate_cost_missing_config_dir_errors(self):
        with patch.object(sys, "argv", ["main.py", "estimate-cost"]):
            with pytest.raises(SystemExit):
                parse_cli_args()

    def test_estimate_cost_with_override(self):
        with patch.object(sys, "argv", [
            "main.py", "estimate-cost", "--config_dir", "configZ_major_v2",
            "--config.world.max_life_steps", "50",
        ]):
            ops, overrides = parse_cli_args()
        assert ops["mode"] == "estimate-cost"
        assert overrides["world.max_life_steps"] == 50


class TestNoSubcommand:
    def test_no_args_errors(self):
        with patch.object(sys, "argv", ["main.py"]):
            with pytest.raises(SystemExit):
                parse_cli_args()


class TestCliSmoke:
    """Smoke tests via subprocess to verify real CLI behavior."""

    def test_help_shows_subcommands(self):
        r = subprocess.run(
            [sys.executable, "main.py", "--help"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "run" in r.stdout
        assert "resume" in r.stdout
        assert "list-runs" in r.stdout
        assert "estimate-cost" in r.stdout

    def test_run_help_shows_config_dir(self):
        r = subprocess.run(
            [sys.executable, "main.py", "run", "--help"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "--config_dir" in r.stdout

    def test_no_args_exits_error(self):
        r = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True, text=True,
        )
        assert r.returncode != 0
