"""
Here we put all system-dependent constants
"""
import numpy as np
import os

LIBRARY_PATH = os.path.dirname((os.path.abspath(__file__)))
THIRDPARTY_PATH = os.path.join(LIBRARY_PATH, '..', 'thirdparty')

# Check for QuickBB
_quickbb_path = os.path.join(
    THIRDPARTY_PATH, 'quickbb', 'quickbb_64')
if os.path.isfile(_quickbb_path):
    QUICKBB_COMMAND = _quickbb_path
else:
    QUICKBB_COMMAND = None

# Check for Tamaki solver
_tamaki_solver_path = os.path.join(
    THIRDPARTY_PATH, 'pace2017_solvers', 'tamaki_treewidth')
if os.path.isdir(_tamaki_solver_path):
    TAMAKI_SOLVER_PATH = _tamaki_solver_path
else:
    raise
    TAMAKI_SOLVER_PATH = None
