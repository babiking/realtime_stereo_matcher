import logging
'''
    Author: https://alexandra-zaharia.github.io/posts/make-your-own-custom-color-formatter-with-python-logging/
'''


class MyFormatter(logging.Formatter):
    """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""
    def __init__(self):
        super().__init__()
        self.grey = '\x1b[38;21m'
        self.blue = '\x1b[38;5;39m'
        self.yellow = '\x1b[38;5;226m'
        self.red = '\x1b[38;5;196m'
        self.bold_red = '\x1b[31;1m'
        self.green = '\x1b[38;5;46m'
        self.orange = '\x1b[38;5;214m'
        self.reset = '\x1b[0m'

        self.level_colors = {
            logging.DEBUG: self.grey,
            logging.INFO: self.blue,
            logging.WARNING: self.yellow,
            logging.ERROR: self.red,
            logging.CRITICAL: self.bold_red
        }

        self.level_formats = {}
        for level_name, level_color in self.level_colors.items():
            self.level_formats[level_name] = ' | '.join([
                level_color + '[%(levelname)s]' + self.reset,
                self.green + '%(asctime)s' + self.reset, \
                self.orange + '%(filename)s -> LINE#%(lineno)d' + self.reset,
                level_color + '%(message)s' + self.reset,
            ])

    def format(self, record):
        log_format = self.level_formats.get(record.levelno)
        formatter = logging.Formatter(log_format)
        return formatter.format(record)


def create_color_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(MyFormatter())
    logger.addHandler(stdout_handler)
    return logger