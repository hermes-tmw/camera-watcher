from . import model
from . import connection

from .model import *
from .connection import *
from .remote import *
from .outdoors import *

__all__ = (model.__all__ + connection.__all__ + remote.__all__ + outdoors.__all__)

import logging
import sys
from pathlib import Path

logname = 'watcher'
logger = None

def setup_logging():
    global logger

    if logger != None:
        return logger

    log_level = (application_config('log', 'LEVEL') or 'INFO').upper()
    assert log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], f"invalid log level {log_level}"

    logger = logging.getLogger(logname)
    logger.setLevel(log_level)

    formatter = logging.Formatter('[%(asctime)s] %(name)s %(levelname)s: %(message)s')

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(log_level)
    consoleHandler.setStream(sys.stdout)
    logger.addHandler(consoleHandler)

    log_path = application_config('log', 'FILE') or 'log/watcher/watcher.log'
    logfile = Path(log_path)
    try:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fileHandler = logging.FileHandler(logfile)
        fileHandler.setLevel(log_level)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
    except OSError:
        logger.warning(f"Could not open log file {logfile}, logging to stdout only")

    return logger

