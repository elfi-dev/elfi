# -*- coding: utf-8 -*-

from elfi.model.elfi_model import *
from elfi.methods.methods import *
from elfi.client import get as get_client
from elfi.store import FileStore
# Use the native client as default
import elfi.clients.native


__author__ = 'ELFI authors'
__email__ = 'elfi-support@hiit.fi'

# make sure __version_ is on the last non-empty line (read by setup.py)
__version__ = '0.5.0'