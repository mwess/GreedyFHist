"""
General utils files. Lowest level of utils. Cannot import from anywhere else in the project.
"""
import shlex
import subprocess
from typing import Dict

import pandas


def call_command(cmd: str):
    """
    Simple wrapper function around a command.
    :param cmd:
    :return:
    """
    ret = subprocess.run(shlex.split(cmd), capture_output=True)
    return ret


def build_cmd_string(path_to_exec: str, args: Dict[str, str]) -> str:
    """Builds a string that can be executed on the command line. 

    Args:
        path_to_exec (str): Path to executable at the beginning of the command.
        args (Dict[str, str]): List of key/value pairs that make of arguments of the command.

    Returns:
        str: command line instruction
    """
    cmd = [path_to_exec]
    for key in args:
        cmd.append(key)
        val = args[key]
        if val != '':
            if isinstance(val, list):
                cmd += val
            else:
                cmd.append(val)
    cmd = ' '.join(cmd)
    return cmd


def scale_table(df: pandas.DataFrame, factor: float) -> pandas.DataFrame:
    """Scales x and y columns according to supplied factor.

    Args:
        df (pandas.DataFrame): 
        factor (float): 

    Returns:
        pandas.DataFrame: scaled dataframe.
    """
    df.x = df.x / factor
    df.y = df.y / factor
    return df
