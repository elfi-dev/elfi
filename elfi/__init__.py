# -*- coding: utf-8 -*-

import elfi.clients.native

import elfi.methods.mcmc
import elfi.model.tools as tools
from elfi.client import get_client, set_client
from elfi.methods.parameter_inference import *
from elfi.methods.post_processing import adjust_posterior
from elfi.model.elfi_model import *
from elfi.model.extensions import ScipyLikeDistribution as Distribution
from elfi.store import OutputPool, ArrayPool
from elfi.visualization.visualization import nx_draw as draw
from elfi.methods.bo.gpy_regression import GPyRegression

__author__ = 'ELFI authors'
__email__ = 'elfi-support@hiit.fi'

# make sure __version_ is on the last non-empty line (read by setup.py)
__version__ = '0.6.1'
