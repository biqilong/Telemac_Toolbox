import numpy as np
import math
import os
import Reading_TelemacData as rt

Test = rt.Reading_TelemacData(2)
d = Test.run()

#slf_dir   = 'C:\\Users\\saaad264\\Research\\Telemac-iFlow\\Telemac_model\\rectangular_mesh\\'
slf_dir   = 'D:\\QILONG_BI\\Research\\Telemac-iFlow\\Telemac_model\\rectangular_mesh\\'
inputfile ='r2D_rectangular_mesh.slf'
slf_file  = slf_dir+inputfile

pp_dir    = slf_dir+'postProcessedData\\'
fig_dir   = slf_dir+'figures\\'

# timestep for extracting 2D results at certain time step
timestep = 2000
# node_nr for extracting time series at certain node
node_nr = 2562  # starting from 0, telemac node nr -1

# the thalweg starts from 2563 to 3843 in the rectangualr mesh
ip_start = 2563
ip_end = 3843

# beginning and final step for time series used in harmonic analysis, starting from 0
tf_start = 740
tf_end = 5733

# read the result info
slf, varNames, id_u, id_v, id_wl, id_ustar, time, XY = Test.TelemacReadHeader(slf_file)

# extract 2D field of a variable
grid_x, grid_y, varField, grid_Var = Test.TelemacReadData2D(slf, XY, timestep, id_ustar)
Test.TelemacPlotFlowfields(fig_dir, timestep, grid_x, grid_y, grid_Var, varNames, id_ustar)
Test.TelemacPlotFlowUV(fig_dir, timestep, grid_x, grid_y, id_u, id_v, XY, varField)

# extract time series of all the variables at one node
timeseries = Test.TelemacReadDataTS(slf, node_nr)
Test.PlotTelemacTimeseries(fig_dir, time, timeseries, node_nr, id_wl, id_u, id_ustar)

# compute M2 and M4 using t-tide
M2_thalweg_WL, M4_thalweg_WL, M2_thalweg_U, M4_thalweg_U = Test.CalculateM2M4(slf, XY, id_wl, id_u, ip_start, ip_end, tf_start, tf_end, pp_dir)
Test.PlotTelemacM2M4(fig_dir, ip_start, ip_end, XY, M2_thalweg_WL, M4_thalweg_WL, M2_thalweg_U, M4_thalweg_U)




