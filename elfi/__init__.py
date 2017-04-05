# -*- coding: utf-8 -*-

import elfi.clients.native
import elfi.model.tools as tools

from elfi.client import get_client, set_client
from elfi.methods.methods import *
from elfi.model.elfi_model import *
from elfi.store import OutputPool
from elfi.visualization.visualization import nx_draw as draw
from elfi.model.extensions import ScipyLikeDistribution as Distribution

import elfi.tools as tools

# Use the native client as default
import elfi.clients.native
import elfi.mcmc

__author__ = 'ELFI authors'
__email__ = 'elfi-support@hiit.fi'

# make sure __version_ is on the last non-empty line (read by setup.py)
__version__ = '0.5.0'

