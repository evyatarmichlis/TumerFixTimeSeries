import logging
import os
from datetime import datetime


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    def __init__(self):
        now = datetime.now()
        datetime_string = now.strftime('D%y%m%dT%H%M')
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'logs')
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            print(f'Created a new log dir in {logs_dir}')
        log_file_path = os.path.join(logs_dir, f'{datetime_string}.log')
        logging.basicConfig(filename=log_file_path,
                            filemode='w',
                            format='%(asctime)s - %(name)s - %(levelname)s\n%(message)s',
                            level=logging.DEBUG)


def _print_and_log(*args, logging_type='info', **kwargs):
    Logger()
    allowed_logging_type = ['info', 'warning', 'error']
    if logging_type not in allowed_logging_type:
        raise ValueError(f'logging_type must be one of {allowed_logging_type} but got logging_type={logging_type}')
    print(*args, **kwargs)
    log_message = ' '.join(map(str, args))
    getattr(logging, logging_type)(log_message)  # For example, logger.info(log_message)


def print_and_log(*args, logging_type='info', run_once: bool = False, **kwargs):
    if run_once:
        if not print_and_log.ran:
            _print_and_log(*args, logging_type=logging_type, **kwargs)
            print_and_log.ran = True
    else:
        _print_and_log(*args, logging_type=logging_type, **kwargs)


print_and_log.ran = False
