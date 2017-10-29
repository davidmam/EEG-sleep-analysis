#!/bin/bash
refDir='/home/vkatsageorgiou/documents/experiments/mouseSleepTucci/sci/baseline2recovery/analysis/epoch4500/'

normFlag=0

python2 classificationEsnemble.py -refDir ${refDir} -normFlag ${normFlag}
