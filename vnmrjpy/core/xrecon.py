import vnmrjpy as vj
import numpy as np

"""
Functions for interfacing with Xrecon 

Interfacing is done in a 'hacked' way. A shell call is made for Xrecon on the
desired fid directory of the desired sequence, a reconstruction is made to fdf
in a temp directory, then fdf data is read into memory again. Fdf files are
optionally deleted. 

Xrecon reconstruction as option can be set in to_kspace(method='xrecon') or
to_imagespace(method='xrecon') methods. It is suggested to set load_data=False
in read_fid() function beforehand.
"""

def xrecon(fid, data='kspace',space='anatomical'):
    """Read fid into a varray with Xrecon reconstruction

    """
    pass

def check_procpar(output='kspace'):
    """Rewrite processing parameters to get the desired output from xrecon

    Args:
        output -- desired xrecon output, either 'kspace' or 'imagespace'
    """ 

    pass

def call(path='default'):
    """Shell call for xrecon"""
    pass

def read_fdf(delete_files=True, path='default'):
    """Read fdf output files of xrecon and put into varray.data""" 
    pass
