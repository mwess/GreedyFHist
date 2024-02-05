"""
General utils files. Lowest level of utils. Cannot import from anywhere else in the project.
"""
import shlex
import subprocess


def call_command(cmd: str):
    """
    Simple wrapper function around a command.
    :param cmd:
    :return:
    """
    ret = subprocess.run(shlex.split(cmd), capture_output=True)
    return ret


def build_cmd_string(path_to_exec, args):
    """Small custom function for collection arguments in a function call."""
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


def scale_table(df, factor):
    df.x = df.x / factor
    df.y = df.y / factor
    return df
