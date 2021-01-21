import logging
import logging.handlers as handlers
import sys
from typing import Dict, Optional
from contextlib import contextmanager
import colorama
from colorama import Back, Fore, Style

from popsynth.utils.configuration import popsynth_config
from popsynth.utils.package_data import get_path_of_log_dir, get_path_of_log_file

colorama.deinit()
colorama.init(strip=False)
## set up the console logging


class ColoredFormatter(logging.Formatter):
    """
    Colored log formatter.
    """
    def __init__(self,
                 *args,
                 colors: Optional[Dict[str, str]] = None,
                 **kwargs) -> None:
        """Initialize the formatter with specified format strings."""

        super().__init__(*args, **kwargs)

        self.colors = colors if colors else {}

    def format(self, record) -> str:
        """Format the specified record as text."""

        record.color = self.colors.get(record.levelname, "")
        record.reset = Style.RESET_ALL

        return super().format(record)


class MyFilter(object):
    def __init__(self, level):
        self.__level = level

    def filter(self, logRecord):
        return logRecord.levelno != self.__level


# now create the developer handler that rotates every day and keeps
# 10 days worth of backup
popsynth_dev_log_handler = handlers.TimedRotatingFileHandler(
    get_path_of_log_file("dev.log"), when="D", interval=1, backupCount=10)

# lots of info written out

_dev_formatter = logging.Formatter(
    "%(asctime)s | %(name)s | %(levelname)s| %(funcName)s | %(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

popsynth_dev_log_handler.setFormatter(_dev_formatter)
popsynth_dev_log_handler.setLevel(logging.DEBUG)

# now set up the usr log which will save the info

popsynth_usr_log_handler = handlers.TimedRotatingFileHandler(
    get_path_of_log_file("usr.log"), when="D", interval=1, backupCount=10)

popsynth_usr_log_handler.setLevel(popsynth_config["logging"]["file"]["level"])

# lots of info written out
_usr_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                   datefmt="%Y-%m-%d %H:%M:%S")

popsynth_usr_log_handler.setFormatter(_usr_formatter)

# now set up the console logger

_console_formatter = ColoredFormatter(
    "{color} {levelname:8} {reset}| {color} {message} {reset}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    colors={
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN + Style.BRIGHT,
        "WARNING": Fore.YELLOW + Style.DIM,
        "ERROR": Fore.RED + Style.BRIGHT,
        "CRITICAL": Fore.RED + Back.WHITE + Style.BRIGHT,
    },
)

popsynth_console_log_handler = logging.StreamHandler(sys.stdout)
popsynth_console_log_handler.setFormatter(_console_formatter)
popsynth_console_log_handler.setLevel(
    popsynth_config["logging"]["console"]["level"])

warning_filter = MyFilter(logging.WARNING)


def silence_warnings():
    """
    supress warning messages in console and file usr logs
    """

    popsynth_usr_log_handler.addFilter(warning_filter)
    popsynth_console_log_handler.addFilter(warning_filter)


def activate_warnings():
    """
    supress warning messages in console and file usr logs
    """

    popsynth_usr_log_handler.removeFilter(warning_filter)
    popsynth_console_log_handler.removeFilter(warning_filter)


def update_logging_level(level):

    popsynth_console_log_handler.setLevel(level)


@contextmanager
def silence_console_log():

    current_console_logging_level = popsynth_console_log_handler.level
    current_usr_logging_level = popsynth_usr_log_handler.level

    popsynth_console_log_handler.setLevel(logging.ERROR)
    popsynth_usr_log_handler.setLevel(logging.ERROR)

    try:
        yield

    finally:

        popsynth_console_log_handler.setLevel(current_console_logging_level)
        popsynth_usr_log_handler.setLevel(current_usr_logging_level)



def show_progress():

    popsynth_config["show_progress"] = True

def silence_progress():

    popsynth_config["show_progress"] = False


        
def setup_logger(name):

    # A logger with name name will be created
    # and then add it to the print stream
    log = logging.getLogger(name)

    # this must be set to allow debug messages through
    log.setLevel(logging.DEBUG)

    # add the handlers

    if popsynth_config["logging"]["debug"]:
        log.addHandler(popsynth_dev_log_handler)

    if popsynth_config["logging"]["console"]["on"]:

        log.addHandler(popsynth_console_log_handler)

    if popsynth_config["logging"]["file"]["on"]:
        log.addHandler(popsynth_usr_log_handler)

    # we do not want to duplicate teh messages in the parents
    log.propagate = False

    return log
