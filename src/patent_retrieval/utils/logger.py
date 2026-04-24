# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.

import logging
import sys

import rich.console
from rich.logging import RichHandler


console_terminal = rich.console.Console()
console_file = None


def get_logger(name=__name__) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        terminal_handler = RichHandler(console=console_terminal, enable_link_path=False)
        terminal_formatter = logging.Formatter("%(message)s")
        terminal_handler.setFormatter(terminal_formatter)
        logger.addHandler(terminal_handler)
        
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class redirect_stdout_stderr:

    def __init__(self, fp):
        self.stdout_original = sys.stdout
        self.stderr_original = sys.stderr
        self.fp = fp
        global console_file
        console_file = rich.console.Console(file=self.fp)

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.stdout_original
        sys.stderr = self.stderr_original

    def write(self, data):
        self.stdout_original.write(data)
        self.stdout_original.flush()
        assert console_file is not None
        console_file.print(data.rstrip())

    def flush(self):
        self.stdout_original.flush()
