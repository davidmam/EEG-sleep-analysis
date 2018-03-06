#
#   THIS CODE IS UNDER THE BSD 2-Clause LICENSE. YOU CAN FIND THE COMPLETE 
#					FILE AT THE SOURCE DIRECTORY.
#				
#       Copyright (C) 2017 V.-M. Katsageorgiou - All rights reserved
#	
#           	@author : vasiliki.katsageorgiou@gmail.com
#
#	
#						Publication:
#	A Novel Unsupervised Analysis of Electrophysiological
#		Signals Reveals New Sleep Sub-stages in Mice
#		
#		
#*****************************************************************************



#!/bin/bash
BASE_DIR="/home/vkatsageorgiou/mouseSleepAnalysis/experiment2/"

# training epochID
epochID=9999

# degrees of polynomial, i.e. deg = 8 for cd1c57mixed, Zfhx3
deg=8

# strains: "cd1c57mixed" OR "Zfhx3"
strains="Zfhx3"

python2 -W ignore lstatesDailyProfiles.py -f ${BASE_DIR} -epoch ${epochID} -case ${strains} -deg ${deg}
