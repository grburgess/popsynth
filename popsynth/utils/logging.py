import logging
import logging.handlers as handlers
import sys
from contextlib import contextmanager
from typing import Dict, Optional

import colorama
from colorama import Back, Fore, Style

from popsynth.utils.configuration import popsynth_config
from popsynth.utils.package_data import get_path_of_log_dir, get_path_of_log_file

colorama.deinit()
colorama.init(strip=False)
# set up the console logging


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


class LoggingState(object):

    def __init__(
        self,
        popsynth_usr_log_handler,
        popsynth_console_log_handler,
    ):
        """
        A container to store the state of the logs.
        """

        # attach the log handlers

        self.popsynth_usr_log_handler = popsynth_usr_log_handler
        self.popsynth_console_log_handler = popsynth_console_log_handler

        # store their current states

        self.popsynth_usr_log_handler_state = popsynth_usr_log_handler.level
        self.popsynth_console_log_handler_state = popsynth_console_log_handler.level

    def _store_state(self):

        self.popsynth_usr_log_handler_state = popsynth_usr_log_handler.level
        self.popsynth_console_log_handler_state = popsynth_console_log_handler.level

    def restore_last_state(self):

        self.popsynth_usr_log_handler.setLevel(
            self.popsynth_usr_log_handler_state)
        self.popsynth_console_log_handler.setLevel(
            self.popsynth_console_log_handler_state)

    def silence_logs(self):

        # store the state
        self._store_state()

        # silence the logs
        self.popsynth_usr_log_handler.setLevel(logging.CRITICAL)
        self.popsynth_console_log_handler.setLevel(logging.CRITICAL)

    def loud_logs(self):

        # store the state
        self._store_state()

        # silence the logs

        self.popsynth_usr_log_handler.setLevel(logging.INFO)
        self.popsynth_console_log_handler.setLevel(logging.INFO)

    def debug_logs(self):

        # store the state
        self._store_state()

        # silence the logs
        self.popsynth_console_log_handler.setLevel(logging.DEBUG)


_log_state = LoggingState(
    popsynth_usr_log_handler,
    popsynth_console_log_handler,
)


def silence_warnings():
    """
    Supress warning messages in console and file usr logs.
    """

    popsynth_usr_log_handler.addFilter(warning_filter)
    popsynth_console_log_handler.addFilter(warning_filter)


def activate_warnings():
    """
    Supress warning messages in console and file usr logs.
    """

    popsynth_usr_log_handler.removeFilter(warning_filter)
    popsynth_console_log_handler.removeFilter(warning_filter)


def update_logging_level(level):

    popsynth_console_log_handler.setLevel(level)


def silence_logs():
    """
    Turn off all logging.
    """

    # handle dev logs independently
    popsynth_dev_log_handler.setLevel(logging.CRITICAL)

    _log_state.silence_logs()


def show_progress_bars():

    popsynth_config.show_progress = True


def silence_progress_bars():

    popsynth_config.show_progress = False


def quiet_mode():
    """
    Turn off all logging and progress bars.
    """

    silence_progress_bars()

    # save state and silence
    silence_logs()


def loud_mode():
    """
    Turn on all progress bars and logging.
    """

    show_progress_bars()

    # save state and make loud
    _log_state.loud_logs()


def activate_logs():
    """
    Re-activate silenced logs.
    """

    # handle dev logs independently
    popsynth_dev_log_handler.setLevel(logging.DEBUG)

    _log_state.restore_last_state()


def debug_mode():
    """
    Activate debug in the console.
    """

    # store state and switch console to debug
    _log_state.debug_logs()


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


def setup_logger(name):
    """
    Set up a new logger.

    :param name: Name of the logger
    """

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
