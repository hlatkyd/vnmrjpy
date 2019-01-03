"""
Vnmrjpy
=======

"""
import sys
import os

from . import aloha
from . import recon
from . import io
from . import util

# declare for easy testing

dataset = '/home/david/dev/vnmrjpy/dataset'
fids = dataset+'/fids'
fdfs = dataset+'/fdfs'
niftis = dataset+'/niftis'
cs = dataset+'/cs'
pics = dataset+'/testpics'


# Read config file
config = util.ConfigParser().parse()

