import vnmrjpy as vj
import os
import json

"""
Collection of unsorted utility functions. Includes:
    
    vprint -- print if verbose is True
    savepd -- save procpar dictionary to json
    loadpd -- load procpar dictionary from json

"""

def vprint(string):

    if vj.config['verbose']==True:
        print(string)
    else:
        pass


def getpetab(pd):

    pass 

def savepd(pd,out):
    """Save procpar dictionary to json file"""
    # enforce .json 
    if out.endswith('.json'):
        pass
    else:
        out = out+'.json'
    # check basedir:
    basedir = out.rsplit('/',1)[0]
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    with open(out,'w') as openfile:
        jsondata = json.dumps(pd)
        json.dump(pd,openfile)

def loadpd(infile):
    """Load procpar dictionary from json file"""
    with open(infile, 'r') as openfile:
        pd = json.load(openfile)
    return pd
