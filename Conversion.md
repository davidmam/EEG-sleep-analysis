## Translating to Python 3

### Simple steps

    1 Convert all tabs to four spaces
    2 Fix print statements
    3 change import cPickle to import _pickle as cPickle
    4 change import ConfigParser to import configparser
    
### Package installs

Install peakdetect with pip

`pip install peakdetect`

Conda package installs

`conda install -c conda-forge adjusttext`

#### installing cudamat 

`git clone  https://github.com/cudamat/cudamat.git`
`conda install cudatoolkit`
`conda install nose`

Need to install visual C++ from Visual Studio to complile cudamat.
Need to install CUDA toolkit directly 

Download VS 2019 commnity edition

Start VS and open a command prompt 
Navigate to cudamat directory and run `python setup.py install`