import logging
import os
import sys

class ColorFormatter(logging.Formatter):
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
        logging.CRITICAL: bold_red
    }

    def format(self, record):
        color = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(
            f'{color}%(asctime)s - %(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s{self.reset}'
        )
        return formatter.format(record)
    
def setup_logger(name=None):
    logger = logging.getLogger(name)
    level = logging.DEBUG if os.getenv('DEBUG') else logging.INFO
    logger.setLevel(level)
    console_handler = logging.StreamHandler(sys.stdout)

    console_handler.setFormatter(ColorFormatter())
    logger.addHandler(console_handler)

    return logger

