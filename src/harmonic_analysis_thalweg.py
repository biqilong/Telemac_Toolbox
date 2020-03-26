import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import FourrierAnalysis as fa

# read the csv data
csvFileDir = 'J:\\18_043_implementation_of_flocculation_model_in_Telemac\\Idealized_Scheldt_ETM\\EBC_rouse_ws=2.8mms_M=6.0e-5_Q=80m3s_init.SSC_2times\\csv_2D_res\\'

# csvFileName = os.listdir(csvFileDir)
csvFileName = 'TS_WL_0km_to_160km.csv'

csvFile = open(csvFileDir+csvFileName, 'r')
lines = csvFile.readlines()

node_nr = 33 # from 0km to 160km every 5km

tmpData = []

for lineStr in lines:
    lineStr = lineStr.rstrip().split(',')
    lineNum = [float(lineStr[i]) for i in range(node_nr)]
    tmpData.append(lineNum)

# data format: [Distance width bottom]
csvData = np.array(tmpData)
csvFile.close()

# harmonic analysis
DT = 1800     # seconds
NSTEP = 17521 # total number of steps

omega = 2.0*np.arccos(-1.)/(3600.0*12.42) #1.4056343e-4
modeNbList = [5, 10]
order = 3
jx = node_nr

period = 2. * np.arccos(-1.) / omega

time = np.array(range(0,NSTEP))*DT

iPeriod01, iPeriod02, iPeriod03 = fa.definePeriods(time, period)

timeLastPeriod = time[iPeriod02-1:iPeriod03]

variableMatrix = csvData[iPeriod02-1:iPeriod03, :]

d = fa.tidalSignals(timeLastPeriod, variableMatrix,jx, omega, modeNbList,3)

M2_amp = np.abs(d["variable0"][:,1])
M4_amp = np.abs(d["variable1"][:,2])
M2_phs = np.angle(d["variable0"][:,1])
M4_phs = np.angle(d["variable1"][:,2])

a = np.hstack((M2_amp.reshape((node_nr,1)),M2_phs.reshape((node_nr,1)),M4_amp.reshape((node_nr,1)),M4_phs.reshape((node_nr,1))))

with open('M2_M4_tide_fine_grid.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(a)

file.close()
