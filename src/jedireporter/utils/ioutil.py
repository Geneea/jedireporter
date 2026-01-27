# coding=utf-8
"""
I/O utilities for jedireporter.
"""

import gzip
import sys
from io import TextIOWrapper
from pathlib import Path
from typing import Optional, Union, cast


def _gzipped(filename: Union[str, Path]) -> bool:
    return Path(filename).suffix in ('.gz', '.gzip')


def openIn(
        filename: Optional[Union[str, Path]], *,
        encoding: str = None,
        gzipped: bool = None,
        translateNl: bool = None
) -> TextIOWrapper:
    """
    Opens an input file or stdin. Supports gzip decompression.

    :param filename: file to read from. If None, stdin is used.
    :param encoding: character encoding used by the file. Utf-8 by default.
    :param gzipped: flag indicating whether the file is gzipped.
       If None, a file with gz and gzip endings is considered to be compressed.
       If set, the flag trumps the file ending.
    :param translateNl: if true or omitted, all new lines are translated to `\n`.
       If false, newlines are left untranslated (useful for reading CSV files).
       This parameter has no effect on stdin.
    :return: the opened file object
    """
    newline = None if translateNl is False else ''
    encoding = encoding or "utf-8"

    if filename:
        filename = str(filename)
        if gzipped or (gzipped is None and _gzipped(filename)):
            return cast(TextIOWrapper, gzip.open(filename, mode='rt', encoding=encoding, newline=newline))
        else:
            return cast(TextIOWrapper, Path(filename).open(mode='rt', encoding=encoding, newline=newline))
    else:
        if gzipped:
            return cast(TextIOWrapper, gzip.open(sys.stdin.buffer, mode='rt', encoding=encoding, newline=newline))
        else:
            return sys.stdin


def openOut(
        filename: Optional[Union[str, Path]], *,
        encoding: str = None,
        gzipped: bool = None,
        translateNl: bool = None
) -> TextIOWrapper:
    """
    Opens an output file (or stdout). Supports gzip compression.

    :param filename: file to write to. If None, stdout is used.
    :param encoding: character encoding used by the file. Utf-8 by default.
    :param gzipped: flag indicating whether the file should be gzipped.
       If None, a file with gz and gzip endings is considered to be compressed.
       If set, the flag trumps the file ending.
    :param translateNl: if true or omitted, all new lines are translated to `\n`.
       If false, newlines are left untranslated (useful for writing CSV files).
       This parameter has no effect on stdout.
    :return: the opened file object
    """
    newline = None if translateNl is False else ''
    encoding = encoding or "utf-8"

    if filename:
        filename = str(filename)
        if gzipped or (gzipped is None and _gzipped(filename)):
            return cast(TextIOWrapper, gzip.open(filename, mode='wt', encoding=encoding, newline=newline))
        else:
            return cast(TextIOWrapper, Path(filename).open(mode='wt', encoding=encoding, newline=newline))
    else:
        if gzipped:
            return cast(
                TextIOWrapper,
                gzip.open(sys.stdout.buffer, mode='wt', encoding=encoding, newline=newline)
            )
        else:
            return sys.stdout
