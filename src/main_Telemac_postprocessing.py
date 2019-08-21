import numpy as np
import math
import os
import matplotlib.pyplot as plt
from scipy.fftpack import fft,fftfreq,fftshift,rfft,irfft,ifft
import Reading_TelemacData as rt

post_input = 'C:\\Users\\saaad264\\Research\\Telemac-iFlow\\Telemac_model\\rectangular_mesh\\am2_1.0_am4_0.05_h_10.0\\post_input.txt'

Test = rt.Reading_TelemacData(2)
d = Test.run(post_input)

# read post_input
fpi = open(post_input, "r")
contents = fpi.readlines()
fpi.close()

for i in range(len(contents)):
    if (contents[i][0] != '#' and contents[i][0] != '\n'):
        tLines = [txt.strip() for txt in contents[i].split('=')]
        if tLines[0] == 'WORKING DIR':
            working_dir = tLines[1]
        elif tLines[0] == '2D RESULT FILE':
            resfile_2D = tLines[1]
        elif tLines[0] == '3D RESULT FILE':
            resfile_3D = tLines[1]
        elif tLines[0] == 'START TIMEFRAME': # beginning of the time series used in harmonic analysis, starting from 0
            tf_start = int(tLines[1])
        elif tLines[0] == 'END TIMEFRAME': # final step of the time series used in harmonic analysis, starting from 0
            tf_end = int(tLines[1])
        elif tLines[0] == 'TIME INTERVAL':
            time_interval = float(tLines[1])
        elif tLines[0] == 'THALWEG POINTS':
            subtLines = tLines[1].split(':')
            ip_start = int(subtLines[0])
            ip_end = int(subtLines[1])

slf_file_2D = working_dir + resfile_2D
slf_file_3D = working_dir + resfile_3D
pp_dir      = working_dir + 'postProcessedData\\'
fig_dir     = working_dir + 'figures\\'

try:
    os.mkdir(pp_dir)
except OSError:
    print ("Creation of the directory %s failed" % pp_dir)
else:
    print ("Successfully created the directory %s " % pp_dir)
try:
    os.mkdir(fig_dir)
except OSError:
    print ("Creation of the directory %s failed" % fig_dir)
else:
    print ("Successfully created the directory %s " % fig_dir)

# read the result info
slf, varNames, id_u, id_v, id_wl, id_ustar, time, XY = Test.TelemacReadHeader(slf_file_2D)

# timestep for extracting 2D results at certain time step
timestep = 2000
# node_nr for extracting time series at certain node
node_nr = 2562  # starting from 0, telemac node nr -1

# extract 2D field of a variable
grid_x, grid_y, varField, grid_Var = Test.TelemacReadData2D(slf, XY, timestep, id_ustar)
Test.TelemacPlotFlowfields(fig_dir, timestep, grid_x, grid_y, grid_Var, varNames, id_ustar)
Test.TelemacPlotFlowUV(fig_dir, timestep, grid_x, grid_y, id_u, id_v, XY, varField)

# extract time series of all the variables at one node
timeseries = Test.TelemacReadDataTS(slf, node_nr)
Test.PlotTelemacTimeseries(fig_dir, time, timeseries, node_nr, id_wl, id_u, id_ustar)

# compute M2 and M4 using t-tide
#M2_thalweg_WL, M4_thalweg_WL, M2_thalweg_U, M4_thalweg_U = Test.CalculateM2M4(slf, XY, id_wl, id_u, ip_start, ip_end, tf_start, tf_end, pp_dir)
M2_thalweg_WL, M4_thalweg_WL, M6_thalweg_WL, M8_thalweg_WL, M2_thalweg_U, M4_thalweg_U, M6_thalweg_U, M8_thalweg_U = Test.CalculateM2M4_FFT(slf, XY, id_wl, id_u, ip_start, ip_end, tf_start, tf_end, time_interval, pp_dir)
Test.PlotTelemacM2M4(fig_dir, ip_start, ip_end, XY, M2_thalweg_WL, M4_thalweg_WL, M2_thalweg_U, M4_thalweg_U)


