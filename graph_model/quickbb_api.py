"""
This module implements interface to QuickBB program.
QuickBB is quite cranky to its input
"""
from .system_defs import QUICKBB_COMMAND
import subprocess


def run_quickbb(cnffile,
                wait_time=60,
                command=QUICKBB_COMMAND,
                cwd=None,
                extra_args=" --min-fill-ordering"):
    """
    Run QuickBB program and collect its output

    Parameters
    ----------
    cnffile : str
         Path to the QuickBB input file
    wait_time : int, default 60
    command : str, optional
         QuickBB command name
    cwd : str, default None
         Current work directory
    extra_args : str, optional
         Optional commands to QuickBB.
         Default: --min-fill-ordering --time 60
    Returns
    -------
    output : str
         Process output
    """
    if command is None:
        raise ValueError('No QuickBB command given.'
                         ' Did you install QuickBB?')
    sh = command + f" --time {int(wait_time)} "
    sh += f"--cnffile {cnffile} "
    if extra_args is not None:
        sh += extra_args

    process = subprocess.Popen(
        sh.split(), stdout=subprocess.PIPE, cwd=cwd)
    output, error = process.communicate()
    if error:
        print(error)

    return output
