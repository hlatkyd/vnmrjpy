import vnmrjpy as vj
import numpy as np
import csv
import os

def getpetab(infile, is_procpar=None, is_petab=None):
    """Get and parse phase encode table for sequence

    Args:
        procpar (str) -- /path/to/procpar
        petab_file -- /path/to/petable_file

    Return:
        petab (np.ndarray)
    """
    def _check_result_petab(petab,params):

        pass

    def _parse_filename(name):
    
        name = name.rsplit('/')[-1]
        pass
    
    def _determine_input_type(infile):

        pass
        # just check name ...
   
    def _readpetabfile(petabfile):

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
 
    if is_procpar == True and is_petab == None:
        filetype = 'pp'
    elif is_petab == True and is_procpar == None:
        filetype = 'petab'
    else:
        filetype = _determine_input_type(infile)
    
    if filetype == 'petab':

        # TODO check file name consistency
        return _readpetabfile(infile) 

    if filetype == 'pp':

        ppdict = vj.io.ProcparReader(infile).read()
        petabfile_name = ppdict['petable']
        try:
            fullpath = vj.config['tablib_dir']+'/'+petabfile_name
            # TODO check file name consistency
            petab = _readpetabfile(fullpath)
            return petab
        except:
            raise(Exception('could not find petab file'))
