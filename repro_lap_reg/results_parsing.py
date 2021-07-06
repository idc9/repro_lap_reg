# import pandas as pd
from numbers import Number
from os.path import join, isdir
import os
import numpy as np


def ensure_numeric(df):
    """
    Ensures numeric columns of pandas DataFrame are actually numbers.

    # TODO: NaNs
    # TODO: bools
    """
    for col in df.columns:

        # check type of first entry
        x = df[col].values[0]

        # if it is a number, format as number
        if isinstance(x, Number):
            df[col] = df[col].astype(float)

    return df


def list_folder_name(path=None):
    """
    Lists the names of the folders in a directory.
    """
    return [p for p in os.listdir(path) if isdir(join(path, p))]


def parse_name(name):
    """
    Parses a name to return the metadata.
    Names are formatted like

    PARAM1=VALUE1__PARAM2=VALUE2__PARAM3

    (note PARAM3 has no value)

    Parameters
    ----------
    name: str
        Name of the folder.

    Output
    ------
    out: dict
    """
    entries = name.split('__')

    out = {}
    for e in entries:

        if '=' in e:
            k, v = e.split('=')
        else:
            k = e
            v = None
        out[k] = v

    return out


def contains_base_name(string, base_name):
    """
    Checks if the base experiment name is contained in a string. Both names should look like
    MODELNAME__PARAM1=VALUE1__PARAM2=VALUE2...

    Parameters
    -----------
    string: str
        We want to know if string contains base name.

    base_name: str
        We want to know if base_name is contained in string.

    Output
    ------
    is_contained: bool
        True if and only if base_name is contined in string
    """

    assert type(string) == str and type(base_name) == str

    if len(string) < len(base_name):
        return False

    elif string == base_name:
        return True

    else:

        if string[0:len(base_name)] == base_name:
            # need to check if we are part of the way through a parameter or vlaue
            n = len(base_name)
            if string[n] == '=' or string[n: n + 2] == '__':
                return True

            else:
                return False

        else:
            return False


def list_expers_with_param(base_name, param, out_data_dir):
    """
    Lists the experiments corresponding to a given base name.

    Parameters
    ----------
    base_name: str
        Base name of the experiment whose results we are loading.

    param: str

    out_data_dir: str
        Directory where the experiments are saved.

    Output
    ------
    expers: list of strs
        Names of the experiments
    """

    expers = []
    for f in list_folder_name(out_data_dir):
        # if base_name in f and param in parse_name(f):
        if contains_base_name(string=f, base_name=base_name) \
                and param in parse_name(f):
            expers.append(f)

    return expers


def parse_model_name(name):
    """
    Parses a model name formatted like

    MODELNAME__PARAM1=VALUE1__PARAM2=VALUE2

    Parameters
    ----------
    name: str

    Output
    ------
    dict: out
    """
    entries = name.split('__')

    out = {}
    out['model'] = entries[0]

    out.update(parse_name('__'.join(entries[1:])))
    return out


def get_model_info(model_list):
    """

    Parameters
    ----------
    model_list: list of str
        List of model names

    Output
    ------
    metadata, base_to_model

    metadata: dict of dicts

    base_to_model: dict of lists

    """

    all_models = np.unique(model_list)
    metadata = {m: parse_model_name(m) for m in all_models}

    # get all base models
    all_base_models = np.unique([metadata[m]['model']
                                 for m in metadata.keys()])
    all_base_models = np.sort(all_base_models)

    # list of models for each base model
    base_to_model = {bm: [] for bm in all_base_models}
    for m, data in metadata.items():
        base_to_model[data['model']].append(m)

    return metadata, base_to_model
