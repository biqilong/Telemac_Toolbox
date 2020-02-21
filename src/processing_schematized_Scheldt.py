import numpy as np
import math
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import ppmodules.selafin_io_pp as sio
from scipy import interpolate
from scipy.fftpack import fft,fftfreq,fftshift,rfft,irfft,ifft
from Reading_TelemacData3D import *


#--------------------------------------#--------------------------------------#--------------------------------------
# task 1 - plot 2DV field

work_dir = 'J:\\18_043_implementation_of_flocculation_model_in_Telemac\\Idealized_Scheldt_ETM\\'
run_name = 'EBC_rouse_ws=2.8mms_M=6.0e-5_Q=80m3s_init.SSC_2times_finegrid_v2'
resfile = '\\r3d_mean_14075_to_14224.slf'
slf_file = work_dir + run_name + resfile

r3d = Reading_TelemacData3D(slf_file)

timestep = 1
id_var = 5
clevel = 100

# r3d.get_2DV_slice_thalweg(1,4,100)
r3d.slf.readVariables(timestep-1)
varPlane = np.array(r3d.slf.getVarValues())
# find thalweg points
node_thalweg3D = [i for i in range(len(r3d.Y)) if r3d.Y[i]==0]
X_thalweg = np.array([r3d.X[i] for i in range(len(r3d.X)) if r3d.Y[i]==0])
Z_thalweg = varPlane[r3d.id_z,node_thalweg3D]
C = varPlane[id_var,node_thalweg3D]
# triangulation
triang = tri.Triangulation(X_thalweg, Z_thalweg)
# mask the outside elements
node_thalweg_bottom =  [i for i in range(len(r3d.Y)) if (r3d.Y[i]==0 and i<=r3d.NPOIN2D)]
node_thalweg_surface = [i for i in range(len(r3d.Y)) if (r3d.Y[i]==0 and i>r3d.NPOIN2D*(r3d.NPLAN-1))]
f_bottom  = interpolate.interp1d(r3d.X[node_thalweg_bottom],  varPlane[r3d.id_z,node_thalweg_bottom])
f_surface = interpolate.interp1d(r3d.X[node_thalweg_surface], varPlane[r3d.id_z,node_thalweg_surface])
triang.set_mask([(Z_thalweg[t].mean() < f_bottom (X_thalweg[t].mean()))
                or (Z_thalweg[t].mean() > f_surface(X_thalweg[t].mean())) for t in triang.triangles])
# plot the figure
fig, ax = plt.subplots()
tcf = ax.tricontourf(triang, C, vmax=0.15, vmin=0.0, levels=clevel, cmap='jet')
fig.colorbar(tcf)
ax.set_title('The tidally-averaged SSC (g/L)')
ax.set_xlabel('Distance from the estuary mouth (m)')
ax.set_ylabel('Elevation (m)')
fig.set_size_inches(12, 5)
fig.savefig('mean_SSC_'+run_name+'.png', dpi=300)
plt.show()

r3d.slf.close()


#--------------------------------------#--------------------------------------#--------------------------------------
# task 2 - plot thalweg variables

work_dir = 'J:\\18_043_implementation_of_flocculation_model_in_Telemac\\Idealized_Scheldt_ETM\\'
run_name = 'THBC_ws=2.8mms_Q=10m3s'
resfile = '\\r3d_mean_14075_to_14224.slf'
slf_file = work_dir + run_name + resfile

r3d = Reading_TelemacData3D(slf_file)

nplan = r3d.slf.getNPLAN()
npoin = r3d.slf.getNPOIN()
npoin2D = int(npoin/nplan)

# find thalweg points
node_thalweg3D = [i for i in range(npoin2D) if r3d.Y[i]==0]
X_thalweg = np.array([r3d.X[i] for i in range(npoin2D) if r3d.Y[i]==0])

for timestep in range(1,150):

    r3d.slf.readVariables(timestep-1)
    varPlane = np.array(r3d.slf.getVarValues())

    Z_thalweg = varPlane[r3d.id_z,node_thalweg3D]

    plt.plot(X_thalweg,Z_thalweg,'o')

plt.show()

r3d.slf.close()


