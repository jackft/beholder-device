
import configparser
import logging
import logging.handlers
import pathlib

from typing import Union

def _log():
    return logging.getLogger(__name__)


def setup_logging(config_path: pathlib.Path,
                  log_level) -> logging.handlers.TimedRotatingFileHandler:
    parser = configparser.ConfigParser(allow_no_value=True)
    with config_path.open("r") as f:
        parser.read_file(f)

    log_path = pathlib.Path(
        parser.get(
            "recorder",
            "log_path",
            fallback="log"
        )
    )
    log_backup = parser.getint(
        "recorder",
        "log_backup",
        fallback=3
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = _log()
    formatter = logging.Formatter(
        "[%(asctime)s %(process)d] [%(name)s] {%(pathname)s:%(lineno)d} [%(levelname)s] %(message)s"
    )
    handler = logging.handlers.TimedRotatingFileHandler(
        log_path, when="h", interval=1, backupCount=log_backup
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.propagate = True
    root.setLevel(log_level)

    return handler


def pathify(str_or_path: Union[str, pathlib.Path]) -> pathlib.Path:
    return pathlib.Path(str_or_path) if isinstance(str_or_path, str) else str_or_path
