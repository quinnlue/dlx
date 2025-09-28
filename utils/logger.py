import logging
from pathlib import Path
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_logger(name, log_file=None, level=logging.INFO):
    """Setup a logger that writes to *log_file* and the console.
    If *log_file* is not provided it will default to <PROJECT_ROOT>/logs/<name>.log
    where <PROJECT_ROOT> is the repository root two levels above this file (dlx/utils/).
    """
    # Determine logs directory relative to project root (two levels up from this file)
    project_root = Path(__file__).resolve().parents[2]
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    # Default log file path
    if log_file is None:
        log_file = logs_dir / f"{name}.log"
    else:
        log_file = Path(log_file)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers in interactive / multi-import environments
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # Formatters
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

# Initialise default train / validation loggers that other modules can import
train_logger = setup_logger('train')
val_logger = setup_logger('val')
