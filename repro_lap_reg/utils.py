import os
from os.path import join
import numpy as np
from sklearn.utils import check_random_state

from datetime import datetime
from pytz import timezone


def get_seeds(n_seeds, random_state=None):
    """
    Samples a set of seeds.

    Parameters
    ----------
    n_seeds: int
        Number of seeds to generate.

    random_state: None, int
        Metaseed used to generate the seeds.
    """
    rng = check_random_state(random_state)
    # return rng.randint(low=0, high=2**32 - 1, size=n_seeds)
    return np.array([sample_seed(rng=rng) for _ in range(n_seeds)])


def sample_seed(rng):
    return rng.randint(low=0, high=2**32 - 1, size=1).item()


def merge_dicts(a, b, allow_key_overlap=True):
    """
    Returns a dict that merges a and b. Entries of b take priority over entries of a.
    """
    if not allow_key_overlap:
        k = set(a.keys()).intersection(b.keys())
        assert len(k) == 0

    out = {k: v for (k, v) in a.items()}
    out.update(b)
    return out


def join_and_make(*args):
    """
    Gets the path for and makes a directory
    """
    fpath = join(*args)
    os.makedirs(fpath, exist_ok=True)
    return fpath


def all_unique(values):
    """
    Checks if each value is unique.

    Parameters
    ----------
    values: iterable

    Output
    ------
    bool
    """

    unique_values = set()
    for i, v in enumerate(values):
        unique_values.add(v)

    return len(unique_values) == i + 1


def get_datetime(zone='EST'):
    """

    Parameters
    ----------
    zone: str
        The zone argument to pytz.timezone()

    Output
    ------
    current_datetime: str
        The current date/time.
    """
    tz = timezone(zone)
    fmt = '%Y-%m-%d %H:%M:%S %Z'
    return datetime.now(tz).strftime(fmt)


def is_power_of_2(x):
    """
    Checks if a number is a power of 2.

    Parameters
    ----------
    x:
        The number to evaluate

    Output
    ------
    yes: bool
        Whether or not x is a positve power of 2.
    """
    if x < 0:
        return False

    elif np.allclose(x, 0):
        return True

    else:
        return np.allclose(np.log2(x), int(np.log2(x)))
