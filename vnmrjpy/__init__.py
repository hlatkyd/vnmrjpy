"""
Vnmrjpy
=======

"""
import sys
import os
import warnings

# import core functionality

from . import core
from .core import *
# setting global constants not accessible at config
DTYPE='complex64'

# import submodules
from . import aloha
from . import recon
from . import io
from . import util
from . import fit
from . import func
from . import sge

# declare for easy testing

# Read config file
config = util.ConfigParser().parse()

