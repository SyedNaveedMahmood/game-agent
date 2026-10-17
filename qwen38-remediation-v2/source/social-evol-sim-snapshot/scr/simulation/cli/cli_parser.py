"""
CLI Parser Module.

Uses jsonargparse subcommands to separate four mutually exclusive modes
(run, resume, list-runs, estimate-cost) while auto-generating --config.*
override args from the Pydantic Config model.
"""

import os
import typing
from jsonargparse import ArgumentParser
from typing import Tuple, Dict, Any, get_args, get_origin
from scr.utils.logger import get_logger

logger = get_logger(__name__)


def validate_config_dir(config_dir: str) -> str:
    """
    Validate the config directory name.

    Args:
        config_dir (str): Name of the config directory

    Returns:
        str: Validated config directory name

    Raises:
        ValueError: If config directory name is invalid
    """
    if not config_dir or not isinstance(config_dir, str):
        raise ValueError("config_dir must be a non-empty string")

    # Check if config directory exists
    config_path = os.path.join('config', config_dir)
    if not os.path.exists(config_path):
        raise ValueError(f"Config directory not found: {config_path}")

    return config_dir


def _add_config_args_from_model(parser, model_cls, prefix="config", skip_fields=None):
    """Recursively add optional CLI args for all Pydantic model fields.

    Only adds primitive-typed fields (int, float, str, bool). Complex types
    (Dict, List, Literal) are skipped as they are rarely overridden via CLI.
    All args are optional with default=None (override semantics).
    """
    from pydantic import BaseModel

    skip_fields = skip_fields or set()

    for name, field_info in model_cls.model_fields.items():
        if name in skip_fields:
            continue

        arg_name = f"--{prefix}.{name}"
        annotation = field_info.annotation

        # Unwrap Optional[X] → X
        origin = get_origin(annotation)
        if origin is typing.Union:
            non_none = [a for a in get_args(annotation) if a is not type(None)]
            base_type = non_none[0] if non_none else str
        else:
            base_type = annotation

        # Recurse into nested Pydantic models
        if isinstance(base_type, type) and issubclass(base_type, BaseModel):
            _add_config_args_from_model(parser, base_type, f"{prefix}.{name}")
        elif base_type in (int, float, str):
            help_text = field_info.description or ""
            parser.add_argument(arg_name, type=base_type, default=None, help=help_text)
        elif base_type is bool:
            help_text = field_info.description or ""
            parser.add_argument(arg_name, type=bool, default=None, help=help_text)
        # Skip complex types (Dict, List, Literal, etc.)


def _extract_overrides(ns, prefix="") -> Dict[str, Any]:
    """Recursively extract non-None leaf values from a nested Namespace.

    jsonargparse stores dotted args as nested Namespace objects.
    This function walks the tree and collects leaf values that are not None.

    Returns:
        Dict with dotted keys like "world.max_life_steps"
    """
    from jsonargparse import Namespace
    overrides = {}
    if ns is None:
        return overrides
    items = vars(ns) if isinstance(ns, Namespace) else {}
    for key, value in items.items():
        # jsonargparse may prefix keys with zero-width space to avoid
        # collisions with Namespace built-in methods (e.g. 'values')
        clean_key = key.lstrip('\u200b')
        full_key = f"{prefix}.{clean_key}" if prefix else clean_key
        if isinstance(value, Namespace):
            overrides.update(_extract_overrides(value, full_key))
        elif value is not None:
            overrides[full_key] = value
    return overrides


def _add_shared_flags(parser):
    """Register operational flags shared across simulation subcommands."""
    parser.add_argument(
        "--checkpoint_dir", type=str, default="./data",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Enable Rich Live dashboard for real-time monitoring",
    )
    parser.add_argument(
        "--no_db", action="store_true",
        help="Disable all database operations (file-only checkpoints)",
    )
    parser.add_argument(
        "--log_level", type=str, default=None,
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set the logging level (default is warning)",
    )
    parser.add_argument(
        "--debug_responses", action="store_true",
        help="Save raw LLM responses when validation errors occur for debugging",
    )


def parse_cli_args() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Parse command-line arguments for the simulation.

    Uses subcommands: run, resume, list-runs, estimate-cost.

    Returns:
        Tuple of (operational args dict, config overrides dict).
        Config overrides use dotted keys like "world.max_life_steps".
    """
    from scr.models.core.config import Config

    parser = ArgumentParser(
        description="Resource and Morality Simulation"
    )
    subcommands = parser.add_subcommands(dest="subcommand", required=True)

    # ── run: start a fresh simulation ──
    run_p = ArgumentParser()
    run_p.add_argument(
        "--config_dir", type=str, required=True,
        help="Name of the config directory to use (e.g., 'configZ_major_v2')",
    )
    _add_shared_flags(run_p)
    _add_config_args_from_model(run_p, Config, prefix="config", skip_fields={"prompts"})
    subcommands.add_subcommand("run", run_p, help="Start a fresh simulation")

    # ── resume: continue from checkpoint ──
    resume_p = ArgumentParser()
    resume_p.add_argument(
        "run_id", type=str,
        help="Run ID to resume (e.g., '0307-121321')",
    )
    resume_p.add_argument(
        "--time_step", type=int, default=None,
        help="Specific time step to resume from",
    )
    _add_shared_flags(resume_p)
    _add_config_args_from_model(resume_p, Config, prefix="config", skip_fields={"prompts"})
    subcommands.add_subcommand("resume", resume_p, help="Resume from checkpoint")

    # ── list-runs: show available runs ──
    list_p = ArgumentParser()
    subcommands.add_subcommand("list-runs", list_p, help="List all available simulation runs")

    # ── estimate-cost: estimate token usage ──
    est_p = ArgumentParser()
    est_p.add_argument(
        "--config_dir", type=str, required=True,
        help="Name of the config directory to use (e.g., 'configZ_major_v2')",
    )
    _add_config_args_from_model(est_p, Config, prefix="config", skip_fields={"prompts"})
    subcommands.add_subcommand("estimate-cost", est_p, help="Estimate token usage and cost")

    # ── Parse ──
    cfg = parser.parse_args()

    # ── Extract ──
    mode = cfg.subcommand
    # jsonargparse stores subcommand args under cfg.<subcommand_name>
    # Hyphenated names are kept as-is (accessed via getattr)
    sub_ns = getattr(cfg, mode)

    ops = {"mode": mode}
    for key in ("config_dir", "checkpoint_dir", "dashboard", "no_db",
                "log_level", "debug_responses", "run_id", "time_step"):
        ops[key] = getattr(sub_ns, key, None)

    # Config overrides live under sub_ns.config (not top-level cfg.config)
    config_ns = getattr(sub_ns, "config", None)
    config_overrides = _extract_overrides(config_ns)

    # Validate config_dir path exists on disk
    if ops.get("config_dir"):
        try:
            ops["config_dir"] = validate_config_dir(ops["config_dir"])
        except ValueError as e:
            parser.error(str(e))

    return ops, config_overrides
