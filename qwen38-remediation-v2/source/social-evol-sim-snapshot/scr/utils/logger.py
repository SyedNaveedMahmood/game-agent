import logging
from pathlib import Path
from typing import Optional

# Import the custom formatter
from scr.utils.formatter import ModuleColoredFormatter  # Replace with actual import path

# Define the module-specific color mapping as shown earlier
MODULE_COLOR_MAPPING = {
    'scr.models.agent': {'INFO': 'blue'},
    'scr.models.environment': {'INFO': 'magenta'},
    'scr.utils.logger': {'INFO': 'cyan'},
    'scr.simulation.generate_checkpoint': {'INFO': 'green'},
    'scr.simulation.update_checkpoint': {'INFO': 'yellow'},
    'scr.api.client': {'INFO': 'purple'},
    'scr.config': {'INFO': 'orange'},
    'scr.utils.tools': {'INFO': 'light_cyan'},
    'scr.main': {'INFO': 'white'},
    # Add more modules and their desired INFO colors here
}

# Default log level
DEFAULT_LOG_LEVEL = logging.DEBUG

# When True, new loggers skip the console handler (dashboard mode)
_console_suppressed = False

# Define a new logging level
OBSERVATION_LEVEL_NUM = 25  # Choose a number between INFO (20) and WARNING (30)
logging.addLevelName(OBSERVATION_LEVEL_NUM, "OBSERVATION")

def observation(self, message, *args, **kws):
    if self.isEnabledFor(OBSERVATION_LEVEL_NUM):
        self._log(OBSERVATION_LEVEL_NUM, message, args, **kws)

logging.Logger.observation = observation

def get_logger(name: str, log_level: int = DEFAULT_LOG_LEVEL) -> logging.Logger:
    """
    Creates and returns a logger with the specified name.
    If the logger already exists, it returns the existing logger.

    Args:
        name (str): Name of the logger, typically __name__.
        log_level (int): Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING).
                         Defaults to logging.WARNING.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.propagate = False  # Prevent propagation to root logger
    if not logger.hasHandlers():
        logger.setLevel(log_level)  # Set the logger level

        # Formatter for console handler with colors, using the custom formatter
        console_formatter = ModuleColoredFormatter(
            fmt='%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'green',  # Default INFO color
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'bold_red',
                'OBSERVATION': 'blue',
            },
            module_colors=MODULE_COLOR_MAPPING  # Pass the module-specific colors
        )

        # Console Handler with color (skip if dashboard is active)
        if not _console_suppressed:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)  # Console log level
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # Attach JSONL handler if one has been initialised
        if _run_jsonl_handler_ref:
            logger.addHandler(_run_jsonl_handler_ref[-1])

    return logger


def init_run_logger(run_dir: Path) -> None:
    """
    Add a per-run JSONStdlibHandler to every existing ``scr.*`` logger.

    Call this once after the run_id is known so that all log output
    is also written to ``<run_dir>/events.jsonl``.
    """
    from scr.utils.sim_logger import JSONStdlibHandler
    from scr.utils import sim_logger

    # Initialise the sim_logger JSONL writer
    sim_logger.init(run_dir)

    # Create JSONL handler for stdlib loggers
    jsonl_handler = JSONStdlibHandler()
    jsonl_handler.setLevel(logging.DEBUG)

    # Attach to all existing scr.* loggers
    for logger_name in list(logging.root.manager.loggerDict):
        if logger_name.startswith('scr.'):
            lg = logging.getLogger(logger_name)
            # Avoid duplicate JSONL handlers
            if not any(isinstance(h, JSONStdlibHandler) for h in lg.handlers):
                lg.addHandler(jsonl_handler)

    # Store the handler so future loggers created via get_logger() can pick it up
    _run_jsonl_handler_ref.append(jsonl_handler)


# Holds a reference to the JSONL handler for new loggers
_run_jsonl_handler_ref: list = []


def suppress_console_logging() -> None:
    """
    Remove console (StreamHandler) from all existing loggers and prevent
    future loggers from adding one.

    Call this when the dashboard is active so log output doesn't
    interfere with the Rich Live display.  File logging is unaffected.
    """
    global _console_suppressed
    _console_suppressed = True

    for logger_name in list(logging.root.manager.loggerDict):
        if logger_name.startswith('scr.') or logger_name == '__main__':
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    logger.removeHandler(handler)


def set_global_log_level(level: int) -> None:
    """
    Set the log level for all loggers in the application.
    
    Args:
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING)
    """
    # Update root logger
    logging.getLogger().setLevel(level)
    
    # Update all existing loggers
    for logger_name in logging.root.manager.loggerDict:
        if logger_name.startswith('scr.'):
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)
            
            # Update console handlers
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    handler.setLevel(level)
