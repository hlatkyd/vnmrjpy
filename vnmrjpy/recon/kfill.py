import numpy as np
import vnmrjpy as vj

class KspaceCompleter():
    """Finalize kspace. Use ALOHA for completion, or pass if not using cs.

    K-space completion is done in the ALOHA framework. Low rank matrix
    completion parameters are initialized from vnmrjpy.config and procpar
    """
    def __init__(self,kspace,procpar):
        """Init ALOHA 'reconpar' and other params"""
        ppdict = vj.io.ProcparReader(procpar).read()

        if ppdict['pslabel'] == 'ge3d_elliptical':
            self.is_complete = True
        elif 'skipint' in ppdict.keys():
            self.is_complete = False
        else:
            self.is_complete = True

        self.kspace = kspace
        self.ppdict = ppdict
        self.procpar = procpar

    def make(self):
        """Run ALOHA or pass if already complete"""

        if self.is_complete:
            return self.kspace
        else:
            # TODO
            raise(Exception('Not implemented yet'))
            # TODO if reconpar is in procpar
            #rp = self.ppdict['reconpar']
            rp = None

            aloha = vj.aloha.Aloha(self.kspace,self.procpar,reconpar=rp)
            return aloha.recon()
    
    def to_global(self):
        pass            
