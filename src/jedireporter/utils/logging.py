# coding=utf-8
"""
Logging utilities for jedireporter.
"""

import logging
import logging.config
import os
import sys
from typing import Any, Mapping, Optional

import yaml


def addLogArguments(parser, **kwargs):
    """
    Adds --log-config argument for specifying a path to a file with the logging configuration.
    @param parser: The argument parser (argparser) to configure.
    @return: The parser passed in.
    """
    group = parser.add_argument_group('Logging options')
    group.add_argument('--log-config', dest='logConfigFile', type=str, default=None, metavar='<path>',
                       help='path to *.yaml or *.ini file with logging configuration,'
                            ' if specified all other options are ignored',
                       **kwargs)
    group.add_argument('--log-file', dest='logFile', type=str, default=None, metavar='<path>',
                       help='path to a log file that will be attached to the root logger', **kwargs)
    group.add_argument('--log-level', dest='logLevel', type=str,
                       choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL'], default=None,
                       help='logging level', **kwargs)
    group.add_argument('--log-stdout', dest='logStdout', action='store_true',
                       help='send log messages to stdout if --log-file is not specified; otherwise log to stderr',
                       **kwargs)
    return parser


def configureFromArgs(args: Any = None):
    """
    Configures logging framework from command-line arguments.
    @param args: Any object which can be converted to a dictionary and supplies keyword arguments of
        the :method:`configure()` method.
    """
    if isinstance(args, Mapping):
        configure(**args)

    elif hasattr(args, '_asdict'):
        # namedtuple
        configure(**args._asdict())

    elif args is not None:
        configure(
            logConfigFile=getattr(args, 'logConfigFile', None),
            logFile=getattr(args, 'logFile', None),
            logLevel=getattr(args, 'logLevel', None),
            logStdout=getattr(args, 'logStdout', False),
        )

    else:
        configure()


def configure(*, logConfigFile: str = None, logFile: str = None, logLevel: str = None, logStdout: bool = False,
              **kwargs):
    """
    Configures logging framework.
    @param logConfigFile: Path to a config file. The logging configuration can either be stored in *.yaml or *.ini
        file. Please read https://docs.python.org/3/library/logging.config.html for details.
    """
    if logConfigFile:
        ext = logConfigFile.rsplit('.', maxsplit=1)
        if len(ext) == 2 and ext[1] == 'yaml':
            with open(logConfigFile, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            cfg['version'] = 1
            cfg['disable_existing_loggers'] = False
            logging.config.dictConfig(cfg)
        else:
            logging.config.fileConfig(logConfigFile, disable_existing_loggers=False)
    else:
        _defaultConfig(logFile=logFile, logLevel=logLevel, logStdout=logStdout)


def _defaultConfig(*, logFile: str = None, logLevel: str = None, logStdout: bool = False):
    # ROOT logger
    fmt = '%(asctime)s %(levelname)s [%(name)s]: %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    rootLevel = logLevel if logLevel else logging.INFO
    if logFile:
        logging.basicConfig(format=fmt, datefmt=datefmt, level=rootLevel, filename=logFile)
    else:
        logStream = sys.stdout if logStdout else sys.stderr
        logging.basicConfig(format='%(message)s', level=rootLevel, stream=logStream)

    # 'jedireporter' logger
    logger = logging.getLogger('jedireporter')
    # the default ROOT logger level is WARNING, but we want more detailed logging from 'jedireporter' packages
    logger.setLevel(logging.NOTSET if logLevel else logging.DEBUG)
    logger.propagate = 1


def getLogger(package: Optional[str], file: Optional[str]) -> logging.Logger:
    """
    Gets a Logger object with its name set to the fully-qualified name of a python module. This is often the same
    as the module's `__name__` attribute, except when `__name__ == '__main__'`. It will return the root logger
    if neither `package` nor `file` are specified.

    Example: LOG = logutil.getLogger(__package__, __file__)

    @param package: module's package
    @param file: module's file path
    @return: `logging.Logger` instance
    """
    if package and file:
        return logging.getLogger('.'.join([package, os.path.splitext(os.path.basename(file))[0]]))
    elif file:
        return logging.getLogger(os.path.splitext(os.path.basename(file))[0])
    else:
        return logging.getLogger()
