import vnmrjpy as vj
import numpy as np

def _check_90deg(procpar):
    """Make sure all rotations are multiples of 90deg

    Args:
        procpar

    Return:
        True or False

    """

    pass



def to_scanner_space(data, procpar):
    """Transform data to scanner coordinate space by properly swapping axes

    Args:
        data (3,4, or 5D np ndarray) -- input data to transform
        procpar (path/to/file) -- Varian procpar file of data

    Return:
        swapped_data (np.ndarray)
    """

    if _ckeck_90deg(procpar) == False:
        raise(Exception('Only supported with rotations of multiples of 90deg'))
    pass
