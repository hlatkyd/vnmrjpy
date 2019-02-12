import vnmrjpy as vj
import numpy as np
import nibabel as nib

class TestFiberFetcher():

    def __init__(self, sequence):

        # sanity check
        if not sequence in ['mems','gems','angio']:
            raise(Exception("incorrenct sequence input. try 'mems', 'angio', etc.."))


        self.sequence = sequence

    def fetch(self):

        def _get_reconpar():
        
            # use aloha init
            pass
    
        if self.sequence == 'angio':
            pass
