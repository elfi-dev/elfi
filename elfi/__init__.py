# -*- coding: utf-8 -*-
from elfi.core import Transform, Simulator, Summary, Discrepancy
from elfi.distributions import *
from elfi.result import *
from elfi.methods import *
from elfi.storage import *
from elfi.visualization import *
from elfi.inference_task import InferenceTask
from elfi.wrapper import *
from elfi.env import client, inference_task, new_inference_task
from elfi import tools

__author__ = 'ELFI authors'
__email__ = 'elfi-support@hiit.fi'

# make sure __version_ is on the last non-empty line (read by setup.py)
__version__ = '0.3.2.dev0'
