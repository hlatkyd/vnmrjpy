import unittest
import vnmrjpy as vj
import test

"""
Test hierarchy: easy -> hard
If one is not working the higher ups are more likely to fail.

test_config.py
test_readprocpar.py

test_readfid.py
test_readfdf.py
test_writenifti.py
test_kmake.py
test_imake.py

test_lmafit.py
test_admm.py
test_aloha.py


"""

if __name__ == '__main__':

    print(vj.config)
    print(vj.fids)
    print(dir(test))


