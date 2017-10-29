#!/bin/bash
BASE_DIR="/run/media/vkatsageorgiou/data/mouseSleepAnalysis/journalpaper_dsetsAsWebpage/experiment1/mcRBManalysis/"

modelName="ws_fac11_cov11_mean10.mat"

python2 inferStates.py -f ${BASE_DIR} -done "True" -m ${modelName}
