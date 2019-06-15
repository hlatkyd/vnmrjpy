import vnmrjpy as vj
import numpy as np
import os
from shutil import copyfile, which, rmtree
import stat
import subprocess
from tempfile import mkstemp, mkdtemp
"""
Functions for interfacing with Xrecon 

Interfacing is done in a 'hacked' way. The target fid directory is linked
into a temp working directory. A modified procpar is copied
to the temp directory which contains the desired Xrecon output type.
Xrecon is called in the temp dir, then fdf files are created, which are read
into varray.data attribute. 
Thge temp directory is deleted afterwards.

Xrecon reconstruction as option can be set in to_kspace(method='xrecon') or
to_imagespace(method='xrecon') methods. It is suggested to set load_data=False
in read_fid() function beforehand.

Xrecon specific parameters are stored in varray.xrp attribute, which is a
dictionary.

"""

def xrecon_fid(fid, data='kspace',space='anatomical'):
    """Read fid into a varray with Xrecon reconstruction

    """
    pass

def make_temp_dir(varr):
    """Make a temp directory and symbolic links with the fid files"""
    tmp = mkdtemp()
    basename = varr.fid_path.rsplit('/',1)[1]
    tmpfid = tmp + '/' + basename
    xrp = {'tempdir': tmp,'tempfid':tmpfid}
    varr.xrp = xrp
    os.mkdir(tmpfid)
    # link fid
    fid = varr.fid_path+'/fid' 
    fidlink = varr.xrp['tempfid']+'/fid'
    os.symlink(fid,fidlink) 
    # copy others
    #procpar
    procpar = varr.fid_path+'/procpar'
    newprocpar = varr.xrp['tempfid']+'/procpar'
    _fullcopy(procpar, newprocpar)
    #log
    log = varr.fid_path+'/log'
    newlog = varr.xrp['tempfid']+'/log'
    _fullcopy(log, newlog)
    #text
    text = varr.fid_path+'/text'
    newtext = varr.xrp['tempfid']+'/text'
    _fullcopy(text, newtext)
    return varr

def mod_procpar(varr, output='kspace'):
    """Rewrite processing parameters to get the desired output from xrecon

    Args:
        output -- desired xrecon output, either 'kspace' or 'imagespace'
    """
    # change procpar in temp dir
    path = varr.xrp['tempfid']
    if output == 'kspace':
        vj.core.utils.change_procpar(path,['rawIM','rawRE'],['y','y'])
        varr.xrp['space'] = 'kspace'
    if output == 'imagespace':
        vj.core.utils.change_procpar(path,['imIM','imRE'],['y','y'])
        varr.xrp['space'] = 'imagespace'
    # read in new procpar dictionary for consistency
    pd = vj.read_procpar(path+'/procpar')
    varr.pd = pd
    return varr

def call(varr):
    """Shell call for xrecon"""
    # check for Xrecon
    if which('Xrecon') is None:
        raise Exception('Cannot find Xrecon')

    # save current workdir
    curdir = os.getcwd()
    # change workdir to temp
    os.chdir(varr.xrp['tempdir'])
    # check if fid exists
    if not os.path.isdir(varr.fid_path.rsplit('/',1)[1]):
        raise Exception('fid not found in proper temp dir')
    # run Xrecon
    cmd = 'Xrecon '+str(varr.fid_path.rsplit('/',1)[-1])
    os.system(cmd)
    varr.xrp['origdir'] = curdir
    os.chdir(curdir)
    return varr

def loadfdf(varr, data='kspace', save_files=False):
    """Read fdf output files of xrecon and put into varray.data""" 

    # change to working temp dir 
    os.chdir(varr.xrp['tempdir'])
    img_list = os.listdir(os.getcwd())
    if data == 'kspace':
        img_IM = sorted([i for i in img_list if 'rawIM' in i])
        img_RE = sorted([i for i in img_list if 'rawRE' in i])
    elif data == 'imagespace':
        img_IM = sorted([i for i in img_list if 'imIM' in i])
        img_RE = sorted([i for i in img_list if 'imRE' in i])
    dataIM = []
    dataRE = []
    for rcvr in range(len(img_RE)):
        # init data shape
        if rcvr == 0:
            re = vj.read_fdf(img_RE[rcvr]).data
            im = vj.read_fdf(img_IM[rcvr]).data
            fullshape = list(re.shape)+[len(img_IM)]
            fulldata = np.zeros(fullshape,dtype='complex64')
            fulldata[...,0] = np.vectorize(complex)(re,im)
            continue
        re = vj.read_fdf(img_RE[rcvr]).data
        im = vj.read_fdf(img_IM[rcvr]).data
        fulldata[...,rcvr] = np.vectorize(complex)(re,im)

    os.chdir(varr.xrp['origdir'])
    varr.data = fulldata
    
    return varr

def clean(varr):
    """Delete intermediate files"""
    tmpdir = varr.xrp['tempdir']
    rmtree(tmpdir,ignore_errors=True)
    return varr

def _fullcopy(src,dst):
    """Copy, and keep permissions correctly"""
    st = os.stat(src)
    uid, gid = st[stat.ST_UID], st[stat.ST_GID]
    copyfile(src, dst)
    os.chown(dst, uid, gid)
