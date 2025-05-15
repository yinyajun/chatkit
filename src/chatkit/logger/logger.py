from loguru import logger

_logger = None


class Logger:
    def log_info(self, message: str):
        raise NotImplemented()

    def log_error(self, message: str):
        raise NotImplemented()

    def log_debug(self, message: str):
        raise NotImplemented()

    def log_warn(self, message: str):
        raise NotImplemented()


class LocalLogger(Logger):
    def log_info(self, message: str):
        logger.info(message)

    def log_error(self, message: str):
        logger.error(message)

    def log_debug(self, message: str):
        logger.debug(message)

    def log_warn(self, message: str):
        logger.warning(message)


class TenLogger(Logger):
    def __init__(self, ten_env):
        self.ten_env = ten_env

    def log_info(self, message: str):
        self.ten_env.log_info(message)

    def log_error(self, message: str):
        self.ten_env.log_info(message)

    def log_debug(self, message: str):
        self.ten_env.log_debug(message)

    def log_warn(self, message: str):
        self.ten_env.log_warn(message)


def get_logger():
    global _logger
    if _logger is None:
        _logger = LocalLogger()
    return _logger


def init_logger(log: Logger):
    global _logger
    _logger = log
