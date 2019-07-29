import numpy as np
import math
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import ppmodules.selafin_io_pp as sio
import ttide as tt

# Input selafin file
slf_file1 = 'D:\\QILONG_BI\\Research\\MCPBE_flocculation_model\\idealized_Scheldt\\idealized_Scheldt.cas_2019-07-27-11h38min41s\\T3DHYD'
slf_file2 = 'D:\\QILONG_BI\\Research\\MCPBE_flocculation_model\\idealized_Scheldt\\idealized_Scheldt.cas_2019-07-27-12h29min17s\\T3DHYD'

##########################################################
# read the file 1 header
slf1 = sio.ppSELAFIN(slf_file1)
slf1.readHeader()

# get the index of the variables
varNames1 = slf1.getVarNames()
for i in range(0,slf1.NBV1): # from 0 to slf1.NBV1-1
    if varNames1[i]=='VELOCITY U      ':
        id_u = i
    if varNames1[i]=='VELOCITY V      ':
        id_v = i
    if varNames1[i]=='FREE SURFACE    ':
        id_wl = i
    if varNames1[i]=='FRICTION VELOCI ':
        id_ustar = i
    if varNames1[i]=='NCOH SEDIMENT1  ':
        id_sand = i
    if varNames1[i]=='MICROFLOCS      ':
        id_micfloc = i
    if varNames1[i]=='MACROFLOCS      ':
        id_macfloc = i
    if varNames1[i]=='MICFLC IN MACFL ':
        id_micf_in_macf = i

# read time stamps
slf1.readTimes()
time1 = slf1.getTimes()
print('Total number of timesteps in file 1: ' + str(len(time1)) + '\n')

# reaf mesh coordinates
X = slf1.x
Y = slf1.y
XY = np.concatenate((np.reshape(X,(slf1.NPOIN,1)),np.reshape(Y,(slf1.NPOIN,1))),axis=1)

# read the file 2 header
slf2 = sio.ppSELAFIN(slf_file2)
slf2.readHeader()

# get the index of the variables
varNames2 = slf2.getVarNames()
for i in range(0,slf1.NBV1): # from 0 to slf1.NBV1-1
    if varNames2[i]=='VELOCITY U      ':
        id_u = i
    if varNames2[i]=='VELOCITY V      ':
        id_v = i
    if varNames2[i]=='FREE SURFACE    ':
        id_wl = i
    if varNames2[i]=='FRICTION VELOCI ':
        id_ustar = i
    if varNames2[i]=='NCOH SEDIMENT1  ':
        id_sand = i
    if varNames2[i]=='COH SEDIMENT1   ':
        id_mud = i

# read time stamps
slf2.readTimes()
time2 = slf2.getTimes()
print('Total number of timesteps in file 2: ' + str(len(time2)) + '\n')

##########################################################
# find the thalweg points (y=0)
# node_thalweg = telemac node nr -1
node_thalweg = [i for i in range(len(X)) if Y[i]==0]
X_thalweg = [X[i] for i in range(len(X)) if Y[i]==0]
Y_thalweg = np.zeros([len(X_thalweg)])
XY_thalweg = np.hstack((np.reshape(node_thalweg,(len(node_thalweg),1)),np.reshape(X_thalweg,(len(X_thalweg),1)),np.reshape(Y_thalweg,(len(Y_thalweg),1))))

##########################################################
# read variables at desired timestep
# select time step = telemac time step -a
timestep = 606

slf1.readVariables(timestep)
varField1 = slf1.getVarValues() # shape (NBV1,NPOIN)

slf2.readVariables(timestep)
varField2 = slf2.getVarValues() # shape (NBV1,NPOIN)

var11_thalweg = [varField1[id_micfloc,i] for i in node_thalweg]
var12_thalweg = [varField1[id_micf_in_macf,i] for i in node_thalweg]
var13_thalweg = [varField1[id_micf_in_macf,i]+varField1[id_micfloc,i] for i in node_thalweg]

var21_thalweg = [varField2[id_mud,i] for i in node_thalweg]

# plot variable along thalweg
fig,axs = plt.subplots()
axs.plot(X_thalweg,var11_thalweg,'+',X_thalweg,var12_thalweg,'.',X_thalweg,var13_thalweg,'o')
axs.set_xlim(0, 160000)
axs.legend(['microflocs', 'macroflocs','total mud'])
axs.set_xlabel('Distance from Vlissingen (m)')
axs.set_ylabel('Concentration (g/L)')
plt.show()

# plot variable along thalweg
fig,axs = plt.subplots()
axs.plot(X_thalweg,var21_thalweg,'+',X_thalweg,var13_thalweg,'o')
axs.set_xlim(0, 160000)
axs.legend(['without flocculation', 'with flocculation'])
axs.set_xlabel('Distance from Vlissingen (m)')
axs.set_ylabel('Concentration (g/L)')
plt.show()

##########################################################
# select node
node_nr = 4279 # starting from 0, telemac node nr -1

# read time series at the desired node
slf1.readVariablesAtNode(node_nr)
timeseries1 = slf1.getVarValuesAtNode()

# read time series at the desired node
slf2.readVariablesAtNode(node_nr)
timeseries2 = slf2.getVarValuesAtNode()

# Plot free surface, velocity and bed shear stress
fig,axs = plt.subplots(3,1)
axs[0].plot(time1, timeseries1[:,id_wl], time2, timeseries2[:,id_wl])
axs[0].set_xlim(0, time1[-1])
axs[0].set_xlabel('time')
axs[0].set_ylabel('Free surface (m)')
axs[0].grid(True)
axs[1].plot(time1, timeseries1[:,id_u], time2, timeseries2[:,id_u])
axs[1].set_xlim(0, time1[-1])
axs[1].set_xlabel('time')
axs[1].set_ylabel('Velocity U (m/s)')
axs[1].grid(True)
axs[2].plot(time1, np.square(timeseries1[:,id_ustar])*1000.0*timeseries1[:,id_u]/abs(timeseries1[:,id_u]), time2, np.square(timeseries2[:,id_ustar])*1000.0*timeseries2[:,id_u]/abs(timeseries2[:,id_u]))
axs[2].set_xlim(0, time1[-1])
axs[2].set_xlabel('time')
axs[2].set_ylabel('Bed shear stress (Pa)')
axs[2].grid(True)
fig.tight_layout()
plt.show()

# Plot sediment concentrations
fig,axs = plt.subplots(3,1)
axs[0].plot(time1, timeseries1[:,id_micf_in_macf]+timeseries1[:,id_micfloc], time2, timeseries2[:,id_mud])
axs[0].set_xlim(0, time1[-1])
axs[0].set_xlabel('time')
axs[0].set_ylabel('Total mud (g/L)')
axs[0].grid(True)
axs[1].plot(time1, timeseries1[:,id_micfloc])
axs[1].set_xlim(0, time1[-1])
axs[1].set_xlabel('time')
axs[1].set_ylabel('Microflocs (g/L)')
axs[1].grid(True)
axs[2].plot(time1, timeseries1[:,id_micf_in_macf])
axs[2].set_xlim(0, time1[-1])
axs[2].set_xlabel('time')
axs[2].set_ylabel('Macroflocs (g/L)')
axs[2].grid(True)
fig.tight_layout()
plt.show()


for iNode in node_thalweg:
    print('Processing time step',str(iNode))
    slf1.readVariablesAtNode(iNode)
    timeseries1 = slf1.getVarValuesAtNode()
    slf2.readVariablesAtNode(iNode)
    timeseries2 = slf2.getVarValuesAtNode()
    tideconout_CS1 = tt.t_tide(timeseries1[:,id_micfloc][360:]+timeseries1[:,id_micf_in_macf][360:],dt=1/6,constitnames=['M2', 'M4', 'M6', 'M8'],out_style=None)
    tideconout_CS2 = tt.t_tide(timeseries2[:,id_mud][360:],dt=1/6,constitnames=['M2', 'M4', 'M6', 'M8'],out_style=None)
    