import setuptools

with open('README.md','r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='vnmrjpy',
    version='0.1.dev1',
    author='David Hlatky',
    author_email='hlatkydavid@gmail.com',
    url='https://github.com/hlatkydavid/vnmrjpy',
    description='Handle VnmrJ MRI data and recostruction with Python',
    long_description=long_description,
    python_requires='>=3.5',
    packages=setuptools.find_packages(exclude=['dataset','dataset*','vnmrsys']),
)
