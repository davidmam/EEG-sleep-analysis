#!/bin/bash
BASE_DIR="/run/media/vkatsageorgiou/data/mouseSleepAnalysis/journalpaper_dsetsAsWebpage/experiment1/"

if [ ! -f "${BASE_DIR}done" ] 
	then
		echo "starting ${BASE_DIR}"
		python2 trainModel.py -f "${BASE_DIR}" -gpuId 0
		#echo "letting gpu 0 cool down" #-- Uncomment in case you want to run multiple experiments
		#sleep 1800  #-- Uncomment in case you want to run multiple experiments
fi
