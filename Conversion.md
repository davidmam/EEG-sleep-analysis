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

Download VS 2017 commnity edition with the C++ build tools


in cudamat:
edit setup.py. Add at line 95

`                if c.find(',ID=2') >1:
                    cmd[idx]=c[:c.index(',ID=2')]+c[c.index(',ID=2')+5:]
`

Open a command prompt
run the script `"c:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"`

Navigate to cudamat directory and run `python setup.py install`

