"""
read
====

Read data from various formats, including Varian specific ones:
    
Varian: fid, fdf, procpar
General Nifti1: nii, nii.gz

"""
import vnmrjpy as vj
from vnmrjpy.core.utils import vprint
import numpy as np
import os
import glob
import csv

def read_procpar(procpar):
    """Return dictionary of varian parameters from procpar file"""

    with open(procpar,'r') as openpp:
        
        name = []
        value = []
        subtype = []
        basictype = []
        maxvalue = []
        minvalue = []
        # possibilities (as of manual): 0 :'line1', 1: 'line2', 2: 'last'
        line_id = 0 
    
        current_name = 0

        for line_num, line in enumerate(openpp.readlines()):

            if line_id == 0:  # parameter name line
                
                fields = line.rsplit('\n')[0].rsplit(' ')
                current_name = fields[0]
                name.append(fields[0])
                subtype.append(int(fields[1]))
                basictype.append(int(fields[2]))

                # can be 0 (undefined), 1 (real), 2 (string)
                current_basictype = int(fields[2])  
                val_list = []  # if parameter value is a multitude of strings
                line_id = 1
                                
            elif line_id == 1:  # parameter value line

                fields = line.rsplit('\n')[0].rsplit(' ')

                # values are real, and all are on this line
                if current_basictype == 1:  
                    val = [fields[i] for i in range(1,len(fields)-1)]
                    if len(val) == 1:  # don't make list if there is only one value
                        val = val[0]
                    value.append(val)
                    line_id = 2
                # values are strings
                elif current_basictype == 2 and int(fields[0]) == 1:
                    
                    value.append(str(fields[1].rsplit('"')[1]))
                    line_id = 2
                # multiple string values
                elif current_basictype == 2 and int(fields[0]) > 1:
                    
                    val_list.append(fields[1])
                    remaining_values = int(fields[0])-1
                    line_id = 3
                
            elif line_id == 2:

                line_id = 0
        
            elif line_id == 3:  # if values are on multiple lines
                if remaining_values > 1:  # count backwards from remaining values
                    val_list.append(fields[0])
                    remaining_values = remaining_values - 1
                elif remaining_values == 1:
                    val_list.append(fields[0])
                    remaining_values = remaining_values - 1
                    value.append(val_list)
                    line_id = 2
                else:
                    raise(Exception('Error reading procpar, problem with line_id'))
            else:
                raise(Exception('Error reading procpar, problem with line_id'))

    vprint('procpar file {} read succesfully'.format(procpar))

    return dict(zip(name, value))

def read_fid(fid,procpar=None):
    """Handles raw data from Varian spectrometer

    .fid File structure as per Vnmrj manual:
    ===================================
    struct datafilehead
    Used at the beginning of each data file (fid's, spectra, 2D) 

       int     nblocks;      /* number of blocks in file
       int     ntraces;      /* number of traces per block    
       int     np;           /* number of elements per trace    
       int     ebytes;       /* number of bytes per element            
       int     tbytes;       /* number of bytes per trace        
       int     bbytes;       /* number of bytes per block        
       short   vers_id;      /* software version and file_id status bits
       short   status;       /* status of whole file    
       int       nbheaders;     /* number of block headers            

    struct datablockhead
     Each file block contains the following header 

       short   scale;    /* scaling factor        
       short   status;    /* status of data in block      
       short   index;    /* block index              
       short   mode;    /* mode of data in block     
       int       ctcount;    /* ct value for FID        
       float   lpval;    /* F2 left phase in phasefile    
       float   rpval;    /* F2 right phase in phasefile
       float   lvl;        /* F2 level drift correction       
       float   tlt;        /* F2 tilt drift correction      
    """

    def _decode_header(bheader):
        """Return dictionary of fid header from binary header data"""
        hkey_list = ['nblocks','ntraces','np','ebytes','tbytes','bbytes',\
                 'vers_id','status','nbheaders']
        hval_list = []
        h_dict = {}
        if len(bheader) != 32:
            raise(Exception('Incorrect fid header data: not 32 bytes'))
        else: 
            hval_list.append(int.from_bytes(bheader[0:4],byteorder='big'))
            hval_list.append(int.from_bytes(bheader[4:8],byteorder='big'))
            hval_list.append(int.from_bytes(bheader[8:12],byteorder='big'))
            hval_list.append(int.from_bytes(bheader[12:16],byteorder='big'))
            hval_list.append(int.from_bytes(bheader[16:20],byteorder='big'))
            hval_list.append(int.from_bytes(bheader[20:24],byteorder='big'))
            hval_list.append(int.from_bytes(bheader[24:26],byteorder='big'))
            hval_list.append(int.from_bytes(bheader[26:28],byteorder='big'))
            hval_list.append(int.from_bytes(bheader[28:],byteorder='big'))

            for num, i in enumerate(hkey_list):
                h_dict[i] = hval_list[num]
        
        return h_dict

    def _decode_blockhead(blockheader):
        """Return block header dictionary"""
        bh_dict = {}
        bhk_list = ['scale','status','index','mode','ctcount','lpval'\
                    'rpval','lvl','tlt']
        bhv_list = []
        bhv_list.append(int.from_bytes(blockheader[0:2],byteorder='big'))
        bhv_list.append(int.from_bytes(blockheader[2:4],byteorder='big'))
        bhv_list.append(int.from_bytes(blockheader[4:6],byteorder='big'))
        bhv_list.append(int.from_bytes(blockheader[6:8],byteorder='big'))
        bhv_list.append(int.from_bytes(blockheader[8:12],byteorder='big'))
        bhv_list.append(int.from_bytes(blockheader[12:16],byteorder='big'))
        bhv_list.append(int.from_bytes(blockheader[16:20],byteorder='big'))
        bhv_list.append(int.from_bytes(blockheader[20:24],byteorder='big'))
        bhv_list.append(int.from_bytes(blockheader[24:28],byteorder='big'))

        for num, i in enumerate(bhk_list):
            bh_dict[i] = bhv_list[num]        
        return bh_dict

    def _chunks(l, n):
        """Generate n long chunks from list l"""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    # init
    if procpar==None:
    
        if os.path.isdir(fid):
            procpar = str(fid)+'/procpar'
            fid = str(fid)+'/fid'
        else:
            procpar = fid.rsplit('/')[0]+'/procpar'

    # header data should be 32 bytes
    with open(fid,'rb') as openFid:
        binary_data = openFid.read()
        # binary header data 
        bheader = bytearray(binary_data[:32])
        # binary fid data
        bdata = binary_data[32:]

    header_dict = _decode_header(bheader)
    pd = read_procpar(procpar)

    chunk_size = int(header_dict['bbytes'])
    block_list = list(_chunks(bdata,chunk_size))

    if header_dict['bbytes'] == len(block_list[0]):
        #print('blocksize check OK')
        pass

    vprint('reading fid {}'.format(fid))
    vprint('\nfid header :\n')
    vprint(header_dict)

    #------------main iteration through bytearray -----------------

    dim = (int(header_dict['nblocks']),\
            int((header_dict['ntraces'])*int(header_dict['np'])))
    fid_data = np.empty(dim)
    if header_dict['ebytes'] == 4:
        dt = '>f'
    if header_dict['ebytes'] == 2:
        dt = '>i2'
    for k,block in enumerate(block_list): # for each block
        block_header = block[:28] # separate block header
        block_data = block[28:]
        fid_data[k,:] = np.frombuffer(bytearray(block_data),dt)

    # fid data ready, now create varray class
    arr = _get_arrayed_par_length(pd)
    sdim = ['phase', 'read', 'slice', 'time', 'rcvr']
    return  vj.varray(data=fid_data,space=None,pd=pd,fid_header=header_dict,\
                        source='fid',dtype=vj.DTYPE, seqcon=pd['seqcon'],\
                        apptype=pd['apptype'],arrayed_params=arr,vdtype='fid',\
                        sdims = sdim)

def read_fdf(path):
    """Return vnmrjpy.varray from varian .fdf files

    Input can be the whole .img directory (this is preferred) or stand-alone
    .fdf files. Procpar file should be present in directory.

    Args:
        path
    Return:
        varray
    """
    #---------------- auxiliary functions for read method -----------------

    def _preproc_fdf(fdf):

        with open(fdf,'rb') as openFdf:
            fdata = bytearray(openFdf.read())
            nul = fdata.find(b'\x00')
            header = fdata[:nul]
            data = fdata[nul+1:]
        return (header,data)

    # ----------parse fdf header and return into a dictionary --------------

    def _parse_header(header):
        keys_to_parse = sorted(['rank','roi','location','spatial_rank',\
                        'matrix','orientation',\
                        'studyid','gap','pe_size','ro_size',\
                        'pe2_size', 'abscissa',\
                        'storage'])
        to_delete = ('char','float','int')
        header = header.decode('ascii').split('\n')
        header_dict = {}    
        for line in header:             # some formatting of header
            vprint(line)
            for item in to_delete:
                if line.startswith(item):
                    line = line.split(item,1)
                    break
            try:
                line = line[1].lstrip()
                line = line.lstrip('*').rstrip(';')
                if '[]' in line:
                    line = line.replace('[]','')
                if '{' in line:
                    line = line.replace('{','(')
                if '}' in line:
                    line = line.replace('}',')')
                if ' ' in line:
                    line = line.replace(' ','')
                line = line.split('=')
                header_dict[line[0]] = line[1]        
            except:
                continue

        for item in keys_to_parse:
            if item in header_dict.keys():            
                if item == 'abscissa':
                    tempval = header_dict[item][1:-1];''.join(tempval)
                    tempval = tempval.replace('"','')                
                    header_dict[item] = tuple([k \
                                    for k in tempval.split(',')])
                if item == 'matrix':            
                    tempval = header_dict[item][1:-1];''.join(tempval)                
                    header_dict[item] = tuple([int(k) \
                                    for k in tempval.split(',')])
                if item == 'roi':
                    tempval = header_dict[item][1:-1];''.join(tempval)                
                    header_dict[item] = tuple([float(k)\
                                    for k in tempval.split(',')])
                if item == 'ro_size' or item == 'pe_size' \
                                    or item == 'pe2_size':
                    header_dict[item] = int(header_dict[item])
                if item == 'storage':
                    tempval = header_dict[item];''.join(tempval)
                    tempval = tempval.replace('"','')
                    header_dict[item] = str(tempval)
                if item == 'orientation':
                    tempval = header_dict[item][1:-1];''.join(tempval)                
                    header_dict[item] = tuple([float(k)\
                                    for k in tempval.split(',')])
                if item == 'location':
                    tempval = header_dict[item][1:-1];''.join(tempval)                
                    header_dict[item] = tuple([float(k)\
                                    for k in tempval.split(',')])
                if item == 'gap':
                    header_dict[item] = float(header_dict[item])
                if item == 'slices':
                    header_dict[item] = int(header_dict[item])
                if item == 'TR':
                    header_dict[item] = float(header_dict[item])/1000
        return header_dict

    #----------process bynary data based on header--------------------

    def _prepare_data(binary_data, header_dict):

        matrix = header_dict['matrix']

        if header_dict['storage'] == 'float' and \
           header_dict['bits'] == '32':
            dt = np.dtype('float32'); dt = dt.newbyteorder('<')

        else:
            raise(Exception('bits may be incorrectly specified in header data'))

        img_data = np.frombuffer(binary_data, dtype=dt)
        img_data = np.reshape(img_data,matrix)
        return img_data
    #---------------------------------------------------------------------
    #                             main read method
    #----------------------------------------------------------------------
    if os.path.isdir(path):
        vprint('\nMaking varray from fdf :  {} \n'.format(path))
        procpar = str(path)+'/procpar'
        fdf_list = sorted(glob.glob(str(path)+'/*.fdf'))
        (header, data) = _preproc_fdf(fdf_list[0])
        pd = read_procpar(procpar)
    else:
        if not path.endswith('.fdf'):
            warnings.warn('Input does not end with .fdf, path might be incorrect')
        vprint('\nMaking varray from fdf :  {} \n'.format(path))
        procpar = path.rsplit('/')[0]+'/procpar'
        pd = read_procpar(procpar)
        fdf_list =[path]
        (header, data) = _preproc_fdf(path)
    
    header_dict = _parse_header(header)
    # ------------------------process if 3d --------------------
    if header_dict['spatial_rank'] == '"3dfov"':

        full_data = []
        time_concat = []
        time = len([1 for i in fdf_list if 'slab001' in i])
        for i in fdf_list: # only 1 item, but there migh be more in future
                            
            (header, data) = _preproc_fdf(i)
            img_data = _prepare_data(data, header_dict)
            full_data.append(img_data) # full data in one list

        data_array = np.asarray(full_data)

    #----------------------process if 2d-------------------------
    elif header_dict['spatial_rank'] == '"2dfov"':

        full_data = []
        time_concat = []
        time = len([1 for i in fdf_list if 'slice001' in i])

        for i in fdf_list:                
            (header, data) = _preproc_fdf(i)
            img_data = _prepare_data(data,header_dict)
            # expand 2d to 4d
            img_data = np.expand_dims(np.expand_dims(img_data,2),3)
            full_data.append(img_data) # full data in one list
        # make sublists
        slice_list =[full_data[i:i+time] \
                        for i in range(0,len(full_data),time)]
        for aslice in slice_list:
            time_concat.append(np.concatenate(tuple(aslice),axis=3))
        #slice+time concatenated
        slice_time_concat = np.concatenate(time_concat, axis=2)
        data_array = slice_time_concat
    # --------------------process if 1d------------------------
    elif header_dict['spatial_rank'] == '"1dfov"':
        raise(Exception('Not implemented'))
     
    # making vnmrjpy.varray
    arrayed_params = _get_arrayed_par_length(pd) 
    
    varr = vj.varray(data=data_array, fdf_header=header_dict, pd=pd,\
                    dtype=vj.DTYPE, source='fdf', seqcon=pd['seqcon'],\
                    apptype=pd['apptype'],arrayed_params=arrayed_params,\
                    vdtype='image',space=None)
    
    varr.space = None
    varr.set_nifti_header()
    varr.to_local()
    return varr

def _get_arrayed_par_length(pd):
    """Return tuple of (name, length) of arrayed acquisition parameters

    List of possible arrayed parameters should be updated in config
    """
    pars = vj.config['array_pars']
    lengths = []
    for par in pars:
        if type(pd[par]) == list:
            l = len(pd[par])
        else:
            l = 1
        lengths.append(l)
    arrayed_pars = [(pars[i],lengths[i]) for i in range(len(pars))]
    return arrayed_pars

def read_petab(ppdict):
    """Get and parse phase encode table for sequence

    Args:
        ppdict -- procpar dictionary

    Return:
        petab (np.ndarray)
    """
    def _readpetabfile(petabfile):
        """Parses petab file contents into appropriate array"""
        lst = []
        lst_fin = []
        if os.path.exists(petabfile):
            pass
        else:
            raise(Exception('Could not find petab file'))
        with open(petabfile, 'r') as openfile:
            reader = csv.reader(openfile,delimiter='\t')
            for item in reader:
                lst.append(item)
            # removing 't1 ='
            lst = lst[1:]
            for item in lst:
                item_fin = []
                for element in item[:-1]:  # the last one is whitespace
                    element = int(element.replace(" ",""))
                    item_fin.append(element)
                lst_fin.append(item_fin)

        petab = np.array(lst_fin)
        return petab

    try:
        petabfile_name = ppdict['petable']
    except:
        return None
    try:
        fullpath = vj.config['tablib_dir']+'/'+petabfile_name
        # TODO check file name consistency
        return _readpetabfile(fullpath)
    except:
        raise(Exception('could not find petab file'))
