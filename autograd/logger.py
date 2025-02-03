import logging
import os
import sys


class ColorFormatter(logging.Formatter):
    """
    A logging formatter that adds ANSI color codes to log messages based on their severity level.

    This formatter maps each logging level to a specific color for improved readability
    in the console output.

    Attributes:
        grey (str): ANSI escape sequence for grey color.
        yellow (str): ANSI escape sequence for yellow color.
        red (str): ANSI escape sequence for red color.
        bold_red (str): ANSI escape sequence for bold red color.
        cyan (str): ANSI escape sequence for cyan color.
        green (str): ANSI escape sequence for green color.
        reset (str): ANSI escape sequence to reset the color.
        FORMATS (dict): Dictionary mapping log levels to their corresponding ANSI color codes.
    """

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    cyan = "\x1b[36;20m"
    green = "\x1b[32;20m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: cyan,
        logging.INFO: green,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red,
    }

    def format(self, record):
        """
        Format the specified record as a colored log message.

        The method selects a color based on the log record's level and applies it to the formatted message.

        Args:
            record (logging.LogRecord): The log record to be formatted.

        Returns:
            str: The formatted log message with ANSI color codes.
        """
        color = self.FORMATS.get(record.levelno)
        # Create a temporary formatter with the selected color and reset sequence
        formatter = logging.Formatter(
            f"{color}%(asctime)s - %(levelname)s - %(message)s{self.reset}"
        )
        return formatter.format(record)


def setup_logger(name=None):
    """
    Set up and configure a logger with colored console output.

    This function creates a logger with the given name, sets its logging level to DEBUG if the environment variable DEBUG is set; otherwise it defaults to INFO level. It then attaches a console handler with a color formatter, and returns the configured logger.

    Args:
        name (str, optional): The name of the logger. Defaults to None.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    level = logging.DEBUG if os.getenv("DEBUG") else logging.INFO
    logger.setLevel(level)
    console_handler = logging.StreamHandler(sys.stdout)

    # Set the formatter for the console handler to use color formatting
    console_handler.setFormatter(ColorFormatter())
    logger.addHandler(console_handler)

    return logger
