# coding=utf-8
"""
CLI utilities for jedireporter.
"""

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from io import TextIOWrapper
from typing import Optional

from jedireporter.utils.ioutil import openIn, openOut


def addInArgGroup(parser: ArgumentParser, title: str = None, prefix: str = 'input', **kwargs):
    """ Creates an argument group to configure input, including file name, number of lines to read, compression,
        and encoding.
        The resulting group is directly supported by the 'argOpenIn' function.
        Multiple inputs are distinguished via the prefix argument, the 'input' prefix allows omitting the input file
        and using stdin instead.

        :param parser: argparser to add the group to
        :param title:  title of the group (default: 'Input')
        :param prefix: string to distinguish between multiple input groups (default: 'input')
        :param kwargs: any additional keyword arguments sent to the file argument
        :return argument group so that additional arguments can be added to it
    """
    group = parser.add_argument_group(title=title or 'Input')
    if prefix == 'input':
        group.add_argument('-i', '--input-file', type=str,
                           help='File name (omit to use stdin)', **kwargs)
        group.add_argument('--input-maxlines', default=None, type=int, help='Number of lines read from the input')
    else:
        group.add_argument(f'--{prefix}-file', type=str, help='File name', **kwargs)
        group.add_argument(f'--{prefix}-maxlines', default=None, type=int, help='Number of lines read from the input')

    group.add_argument(f'--{prefix}-gz', action='store_true',
                       help='The input is compressed (also triggered by gz/gzip file suffix)')
    group.add_argument(f'--{prefix}-enc', type=str, default='utf-8', help='Input encoding')
    return group


def addOutArgGroup(parser: ArgumentParser, title: str = None, prefix: str = 'output', **kwargs):
    """ Creates an argument group to configure output, including file name, compression, and encoding.
        The resulting group is directly supported by the 'argOpenOut' function.
        Multiple output are distinguished via the prefix argument, the 'output' prefix allows omitting the input file
        and using stdout instead.

        :param parser: argparser to add the group to
        :param title:  title of the group (default: 'Output')
        :param prefix: string to distinguish between multiple output groups (default: 'output')
        :param kwargs: any additional keyword arguments sent to the file argument
        :return argument group so that additional arguments can be added to it
    """
    group = parser.add_argument_group(title=title or 'Output')
    if prefix == 'output':
        group.add_argument('-o', '--output-file', type=str, help='File name (omit to use stdout)', **kwargs)
    else:
        group.add_argument(f'--{prefix}-file', type=str, help='File name', **kwargs)

    group.add_argument(f'--{prefix}-gz', action='store_true',
                       help='The output is compressed (also triggered by gz/gzip file suffix)')
    group.add_argument(f'--{prefix}-enc', type=str, default='utf-8', help='Output encoding')
    return group


def addInOutArgGroups(parser: ArgumentParser, in_title: str = None, out_title: str = None) -> ArgumentParser:
    """ Convenience method adding the default input and output groups. """
    addInArgGroup(parser, title=in_title)
    addOutArgGroup(parser, title=out_title)
    return parser


@dataclass
class ArgIOSpec:
    filename: Optional[str]     # using str because pathlib.Path does not support protocols yet
    encoding: str
    gzipped: Optional[bool]
    """ True or None. The value False (explicitly non-compressed) is not currently used. """
    maxLines: Optional[int] = None

    @staticmethod
    def fromInArgs(args: Namespace, args_prefix: str = None) -> "ArgIOSpec":
        """ Ad input specified via :func:`addInArgGroup`."""
        args_prefix = args_prefix or 'input'
        filename = getattr(args, f'{args_prefix}_file', None)
        assert filename or args_prefix == 'input'
        gzipped = getattr(args, f'{args_prefix}_gz', None)

        return ArgIOSpec(
            filename=filename,
            encoding=getattr(args, f'{args_prefix}_enc', None),
            gzipped=gzipped if gzipped else None,
            maxLines=getattr(args, f'{args_prefix}_maxlines', None)
        )

    @staticmethod
    def fromOutArgs(args: Namespace, args_prefix: str = None) -> "ArgIOSpec":
        """ Ad output specified via :func:`addOutArgGroup`."""
        args_prefix = args_prefix or 'output'
        filename = getattr(args, f'{args_prefix}_file', None)
        assert filename or args_prefix == 'output'
        gzipped = getattr(args, f'{args_prefix}_gz', None)

        return ArgIOSpec(
            filename=filename,
            encoding=getattr(args, f'{args_prefix}_enc', None),
            gzipped=gzipped if gzipped else None,
            maxLines=getattr(args, f'{args_prefix}_maxlines', None)
        )


def argOpenIn(args: Namespace, *, args_prefix: str = None, translateNl: bool = None) -> TextIOWrapper:
    """
    Opens an input file (or stdin) according to the CLI arguments configured by the :func:`addInArgGroup` function.

    :param args: namespace object from ``argparser.parse_args()``
    :param args_prefix: string to distinguish between multiple input groups (default: 'input')
    :param translateNl: if true or omitted, all new lines are translated to `\n`.
       If false, newlines are left untranslated (useful for reading CSV files).
       This parameter has no effect on stdin.
    :return: the opened file
    """
    input = ArgIOSpec.fromInArgs(args, args_prefix=args_prefix)
    return openIn(input.filename, encoding=input.encoding, gzipped=input.gzipped, translateNl=translateNl)


def argOpenOut(args: Namespace, *, args_prefix: str = None, translateNl: bool = None) -> TextIOWrapper:
    """
    Opens an output file (or stdout) according to the CLI arguments configured by the :func:`addOutArgGroup` function.

    :param args: namespace object from ``argparser.parse_args()``
    :param args_prefix: string to distinguish between multiple output groups (default: 'output')
    :param translateNl: if true or omitted, all new lines are translated to `\n`.
       If false, newlines are left untranslated (useful for reading CSV files).
       This parameter has no effect on stdin.
    :return: the opened file
    """
    output = ArgIOSpec.fromOutArgs(args, args_prefix=args_prefix)
    return openOut(output.filename, encoding=output.encoding, gzipped=output.gzipped, translateNl=translateNl)
