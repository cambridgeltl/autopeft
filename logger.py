import logging
import sys


def setup_logging(save_path, mode='a'):
    """
    Sets up the logging_setup for multiple processes. Only enable the logging_setup for the
    master process, and suppress logging_setup for the non-master processes.
    """
    # Enable logging_setup for the master process.
    logging.root.handlers = []

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    print_plain_formatter = logging.Formatter(
        "[%(asctime)s]: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    fh_plain_formatter = logging.Formatter("%(message)s")

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(print_plain_formatter)
    logger.addHandler(ch)

    if save_path is not None:
        fh = logging.FileHandler(save_path, mode=mode)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fh_plain_formatter)
        logger.addHandler(fh)


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)