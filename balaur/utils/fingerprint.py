from io import BytesIO

import dill
import xxhash

from typing import *


def dumps(obj: Any):
    """pickle an object to a string"""
    file = BytesIO()
    dill.Pickler(file, recurse=True).dump(obj)
    return file.getvalue()


def fingerprint_dict(state: Dict[str, Any]):
    m = xxhash.xxh64()
    for key in sorted(state):
        m.update(key)
        m.update(dumps(state[key]))
    return m.hexdigest()



