import vnmrjpy as vj
import numpy as np

def disgustingworkaroundforpossiblebug(p):
    """this is the solution for weird interleaving effects

    p is procpar dictionary
    returns boolean

    use when sliceorder is 1 but is not actually interleaved
    rearrange slices in kmake when false
    """
    def _evenslices(p):
        print('slices {}'.format(p['ns']))
        return (int(p['ns']) % 2 == 0)

    # this means interleaved
    if int(p['sliceorder'])==1:

        # switch for sequnces
        #if p['pslabel'][:4] == 'gems':  
        # switch for seqcon
        if p['seqcon'] == 'nccnn':
            
            # switch for even-odd slices
            if _evenslices(p):

                # switch for orientation
                if p['orient'] in ['sag','sag90','trans90','cor','cor90']:
                    return True

                else:
                    return False

            else:
                # switch for orientation
                if p['orient'] in ['sag','sag90','trans','cor','cor90']:
                    return True
                else:
                    return False
        
        return False
    
    # this means no interleave
    else:
        return False
