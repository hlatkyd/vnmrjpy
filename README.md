## Vnmrjpy

Work-in-progress version, most things are not functioning properly at this stage.

Developmental release is for easier grid engine deployment and testing.

### Intentions

 - Read and write various Vnmrj related formats (fdf, fid) and Nifti.
 - Hold image, acquisition, and reconstruction data consistently in one class 
 - Handle basic K-space reconstruction for most sequences with the same quality as Vnmrj

### In the future

 - Support compressed sensing reconstruction in ALOHA framework
 - Various parameter fitting methods (COMPOSER, WASABI, etc...)
 - Preprocessing with FSL and Nipype

### Install

`pip install vnmrjpy` 

### Some basic usage
Read the fid directory:

`seqdata = vnmrjpy.read_fid('test.fid')`

Transform to k-space, and access as Numpy array:

`seqdata.to_kspace().data`

Transform to image space:

`seqdata.to_imagespace()`
