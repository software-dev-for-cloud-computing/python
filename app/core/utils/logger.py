import contextvars
import logging
from functools import wraps
from typing import Callable, Any

request_id_var = contextvars.ContextVar('request_id', default=None)


class Logger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log(self, level: str, message: str, func_name: str = None):
        request_id = request_id_var.get()
        log_message = f"Request Id: {request_id} | Function: {func_name} | Message: {message}"
        if level == 'debug':
            self.logger.debug(log_message)
        elif level == 'info':
            self.logger.info(log_message)
        elif level == 'warning':
            self.logger.warning(log_message)
        elif level == 'error':
            self.logger.error(log_message)
        elif level == 'critical':
            self.logger.critical(log_message)

    def debug(self, message: str, func_name: str = None):
        self.log('debug', message, func_name)

    def info(self, message: str, func_name: str = None):
        self.log('info', message, func_name)

    def warning(self, message: str, func_name: str = None):
        self.log('warning', message, func_name)

    def error(self, message: str, func_name: str = None):
        self.log('error', message, func_name)

    def critical(self, message: str, func_name: str = None):
        self.log('critical', message, func_name)

    def log_decorator(self, level: str = 'info', message: str = 'Executing'):
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.log(level=level, message=message, func_name=func.__name__)
                result = func(*args, **kwargs)
                return result

            return wrapper

        return decorator
