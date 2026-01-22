# coding=utf-8
"""
Utilities for reading/writing values to nested dictionaries.
"""

import re
from typing import Any, List, Mapping, Optional, Union


class ValueGetter:
    """
    Example:
        data = { 'labels': [{'en': {'value': 'house'}, 'cs': 'dum'}, {'en': 'residence', 'cs': 'sidlo'}]
        getter = ValueGetter('en.value')
        labels = map(getter, data['labels'])
    """
    def __init__(self, path: str, default: Any = None):
        self.__path = path
        self.__default = default

    def get(self, x: Mapping[str, Any]) -> Any:
        return getValue(x, self.__path, self.__default)

    def __call__(self, x: Mapping[str, Any]) -> Any:
        return self.get(x)


def _convertNegativeIndex(x: int, array: list):
    if len(array) == 0 and x == -1:
        return 0    # Allow adding with -1 to the end of the empty list
    elif len(array) < abs(x):
        raise IndexError(f'Cannot append to {x} position, the list has size {len(array)}.')
    else:
        return x % len(array)   # Modulo converts the index to positive one


def getValue(obj: Optional[Union[Mapping[str, Any], List[Any]]], key: str, default: Any = None) -> Any:
    """
    Gets a value from a nested mapping associated with the key. Use a dot character for separating
    each nested mapping that needs to be traversed. Nested lists can be accessed using an index wrapped
    in square brackets, e.g. "key.[0].another_key.[1]".

    @param obj: the mapping object
    @param key: the nested key
    @param default: a value that will be returned if the nested key does not exist in the mapping object
    @return: the value associated with the nested key or the default value
    """
    d = obj
    for k in key.split('.'):
        if match := re.match(r'\[(-?\d+)]', k):
            index = int(match.group(1))
            index = _convertNegativeIndex(index, d) if index < 0 else index
            d = d[index] if isinstance(d, List) and len(d) > index else None
        else:
            d = d.get(k) if isinstance(d, Mapping) else None
        if d is None:
            return default

    return d
