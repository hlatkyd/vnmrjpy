import setuptools
import glob

with open('README.md','r') as fh:
    long_description = fh.read()

script_list = glob.glob('vnmrjpy/scripts/*')

setuptools.setup(
    name='vnmrjpy',
    version='0.1.dev2',
    author='David Hlatky',
    author_email='hlatkydavid@gmail.com',
    url='https://github.com/hlatkydavid/vnmrjpy',
    description='Handle VnmrJ MRI data and recostruction with Python',
    long_description=long_description,
    python_requires='>=3.5',
    #packages=setuptools.find_packages(exclude=\
    #['dataset','test','.test.*','*.test.*',\
    #'vnmrsys','vnmrjpy.bin','vnmrjpy.recon','vnmrjpy.io']),
    packages=['vnmrjpy',
            'vnmrjpy.core',
            'vnmrjpy.fit',
            'vnmrjpy.aloha',
            'vnmrjpy.config',
            'vnmrjpy.func',
            'vnmrjpy.sge',
            'vnmrjpy.util'],
    scripts=script_list
)
