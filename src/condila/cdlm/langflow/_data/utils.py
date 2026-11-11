"""The small subset of dataset helpers used by the LangFlow reproduction."""
import logging

import fsspec


def fsspec_exists(filename):
    filesystem, path = fsspec.core.url_to_fs(filename)
    return filesystem.exists(path)


def fsspec_mkdirs(dirname, exist_ok=True):
    filesystem, path = fsspec.core.url_to_fs(dirname)
    filesystem.makedirs(path, exist_ok=exist_ok)


def get_logger(name=__name__, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

