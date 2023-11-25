import argparse
import logging


def init_logging() -> None:
    """Initializes logging capabilities for entire applicaiton"""

    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('-log')
    args = parser.parse_args()
    log_level = log_levels[args.log] if args.log else logging.ERROR
    logging.basicConfig(level=log_level, filename="logs/log.log",
                        filemode="w", format='%(asctime)s - [%(levelname)s]: %(message)s')