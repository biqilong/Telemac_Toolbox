import numpy as np
import math
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import ppmodules.selafin_io_pp as sio

class Reading_TelemacData3D:
    def __init__(self, slf_file):

        # golobal variables
        self.slf_file = slf_file
        self.slf = []
        self.varNames = []
        self.time = []
        self.id_u = []
        self.id_v = []
        self.id_w = []
        self.id_z = []
        self.X = []
        self.Y = []
        self.Z = []
        self.NPOIN = []
        self.NPOIN2D = []
        self.NPLAN = []
        self.NSTEP = []

        # read the file header
        self.slf = sio.ppSELAFIN(self.slf_file)
        self.slf.readHeader()
        # read mesh data
        self.NPOIN = self.slf.NPOIN
        self.NPLAN = self.slf.NPLAN
        self.NPOIN2D = int(self.NPOIN/self.NPLAN)

        # get the index of the variables
        self.varNames = self.slf.getVarNames()
        # get the variable index
        for i in range(0,self.slf.NBV1): # from 0 to slf.NBV1-1
            if self.varNames[i]=='VELOCITY U      ':
                self.id_u = i
            if self.varNames[i]=='VELOCITY V      ':
                self.id_v = i
            if self.varNames[i]=='VELOCITY W      ':
                self.id_w = i
            if self.varNames[i]=='ELEVATION Z     ':
                self.id_z = i
        # read time stamps
        self.slf.readTimes()
        self.time = self.slf.getTimes()
        self.NSTEP = len(self.time)
        print('Total number of timesteps in the selafin file: ' + str(len(self.time)) + '\n')
        # read mesh coordinates
        self.X = self.slf.x
        self.Y = self.slf.y

    def get_raw_profile(self, IPOIN2D): # 2D node nr
        IPOIN3D = [IPOIN2D+self.NPOIN2D*IPLAN for IPLAN in range(self.NPLAN)]
        raw_profiles = []
        for i in IPOIN3D:
            self.slf.readVariablesAtNode(i-1)
            raw_profiles.append(self.slf.getVarValuesAtNode())
        return IPOIN3D, raw_profiles

    def get_var_profile(self, raw_profiles, id_var, IPOIN3D, start_t, end_t):
        varProfile = []
        n = 0
        for t in range(start_t,end_t+1):
            self.Z.append([raw_profiles[i][t,self.id_z] for i in range((len(IPOIN3D)))])
            varProfile.append([raw_profiles[i][t,id_var] for i in range((len(IPOIN3D)))])
            plt.plot(varProfile[n],self.Z[n])
            n = n+1
        plt.xlabel(self.slf.vars[id_var])
        plt.ylabel('Elevation (m)')
        plt.show()
        return varProfile

    def get_2DV_slice_thalweg(self, timestep, id_var, clevel):
        # read data at timestep
        self.slf.readVariables(timestep-1)
        varPlane = np.array(self.slf.getVarValues())
        # find thalweg points
        node_thalweg3D = [i for i in range(len(self.Y)) if self.Y[i]==0]
        X_thalweg = np.array([self.X[i] for i in range(len(self.X)) if self.Y[i]==0])
        Z_thalweg = varPlane[self.id_z,node_thalweg3D]
        C = varPlane[id_var,node_thalweg3D]
        # triangulation
        triang = tri.Triangulation(X_thalweg, Z_thalweg)
        # mask the outside elements
        node_thalweg_bottom =  [i for i in range(len(self.Y)) if (self.Y[i]==0 and i<=self.NPOIN2D)]
        node_thalweg_surface = [i for i in range(len(self.Y)) if (self.Y[i]==0 and i>self.NPOIN2D*(self.NPLAN-1))]
        f_bottom  = interpolate.interp1d(self.X[node_thalweg_bottom],  varPlane[self.id_z,node_thalweg_bottom])
        f_surface = interpolate.interp1d(self.X[node_thalweg_surface], varPlane[self.id_z,node_thalweg_surface])
        triang.set_mask([(Z_thalweg[t].mean() < f_bottom (X_thalweg[t].mean()))
                      or (Z_thalweg[t].mean() > f_surface(X_thalweg[t].mean())) for t in triang.triangles])
        # plot the figure
        plt.tricontourf(triang, C, levels=clevel)
        #plt.triplot(triang, 'ko-')
        #plt.plot(r3d.X[node_thalweg_bottom],  varPlane[r3d.id_z,node_thalweg_bottom],  'ro')
        #plt.plot(r3d.X[node_thalweg_surface], varPlane[r3d.id_z,node_thalweg_surface], 'bo')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel(self.slf.vunits[id_var])
        plt.show()
        return node_thalweg3D, X_thalweg, Z_thalweg, varPlane

# test
slf_file = 'D:\\QILONG_BI\\Research\\MCPBE_flocculation_model\\idealized_Scheldt\\r3D_idealized_Scheldt_2CPBE.slf'
r3d = Reading_TelemacData3D(slf_file)

IPOIN2D = 5550
start_t = 355
end_t = 429

IPOIN3D, raw_profiles = r3d.get_raw_profile(IPOIN2D)
varProfile = r3d.get_var_profile(raw_profiles, 4, IPOIN3D, start_t, end_t)

#--------------------------------------
timestep = 373
for timestep in range(365,381):
    node_thalweg3D, X_thalweg, Z_thalweg, varPlane = r3d.get_2DV_slice_thalweg(timestep,6,100)

#--------------------------------------
# computing settling flux

t0 = 355
t1 = 429

slf_file = 'D:\\QILONG_BI\\Research\\MCPBE_flocculation_model\\idealized_Scheldt\\r3D_idealized_Scheldt_2CPBE.slf'
r3d = Reading_TelemacData3D(slf_file)
r3d.varNames
# find thalweg points
node_thalweg3D = [i for i in range(len(r3d.Y)) if r3d.Y[i]==0]
X_thalweg = np.array([r3d.X[i] for i in range(len(r3d.X)) if r3d.Y[i]==0])
#Z_thalweg = varPlane[r3d.id_z,node_thalweg3D]
varPlane_total = np.zeros([r3d.slf.NBV1,r3d.slf.NPOIN])
SettlingFlux_2CPBE = np.zeros([len(node_thalweg3D)])
for timestep in range(t0,t1+1):
    r3d.slf.readVariables(timestep)
    varPlane = np.array(r3d.slf.getVarValues())
    varPlane_total = varPlane_total+varPlane
    # C = np.array(varPlane[6,node_thalweg3D])
    # C = np.array((varPlane[8,node_thalweg3D]+varPlane[10,node_thalweg3D])*varPlane[6,node_thalweg3D])
    # C = np.array(varPlane[8,node_thalweg3D]+varPlane[10,node_thalweg3D])
    C = np.multiply((varPlane[8,node_thalweg3D]+varPlane[10,node_thalweg3D]),varPlane[1,node_thalweg3D])
    SettlingFlux_2CPBE = SettlingFlux_2CPBE+C
varPlane_total = varPlane_total/(t1-t0+1)
SettlingFlux_2CPBE = SettlingFlux_2CPBE/(t1-t0+1)
Z_thalweg = varPlane_total[r3d.id_z,node_thalweg3D]


slf_file = 'D:\\QILONG_BI\\Research\\MCPBE_flocculation_model\\idealized_Scheldt\\r3D_idealized_Scheldt.slf'
r3d = Reading_TelemacData3D(slf_file)
r3d.varNames
# find thalweg points
node_thalweg3D = [i for i in range(len(r3d.Y)) if r3d.Y[i]==0]
X_thalweg = np.array([r3d.X[i] for i in range(len(r3d.X)) if r3d.Y[i]==0])
#Z_thalweg = varPlane[r3d.id_z,node_thalweg3D]
varPlane_total = np.zeros([r3d.slf.NBV1,r3d.slf.NPOIN])
SettlingFlux = np.zeros([len(node_thalweg3D)])
for timestep in range(t0,t1+1):
    r3d.slf.readVariables(timestep)
    varPlane = np.array(r3d.slf.getVarValues())
    varPlane_total = varPlane_total+varPlane
    C = np.array((varPlane[8,node_thalweg3D])*0.0037)
    # C =  np.array(varPlane[8,node_thalweg3D])
    # C = np.multiply(varPlane[8,node_thalweg3D],varPlane[1,node_thalweg3D])
    SettlingFlux = SettlingFlux+C
varPlane_total = varPlane_total/(t1-t0+1)
SettlingFlux = SettlingFlux/(t1-t0+1)
Z_thalweg = varPlane_total[r3d.id_z,node_thalweg3D]


# triangulation
triang = tri.Triangulation(X_thalweg, Z_thalweg)
# mask the outside elements
node_thalweg_bottom =  [i for i in range(len(r3d.Y)) if (r3d.Y[i]==0 and i<=r3d.NPOIN2D)]
node_thalweg_surface = [i for i in range(len(r3d.Y)) if (r3d.Y[i]==0 and i>r3d.NPOIN2D*(r3d.NPLAN-1))]
f_bottom  = interpolate.interp1d(r3d.X[node_thalweg_bottom],  varPlane_total[r3d.id_z,node_thalweg_bottom])
f_surface = interpolate.interp1d(r3d.X[node_thalweg_surface], varPlane_total[r3d.id_z,node_thalweg_surface])
triang.set_mask([(Z_thalweg[t].mean() < f_bottom (X_thalweg[t].mean()))
                or (Z_thalweg[t].mean() > f_surface(X_thalweg[t].mean())) for t in triang.triangles])
# plot the figure
plt.tricontourf(triang, SettlingFlux_2CPBE-SettlingFlux, levels=200, cmap='coolwarm')
plt.xlim(5000,155000)
# plt.clim(0.0000,0.3)
plt.clim(-0.02,0.02)
# plt.clim(80,180)
cbar = plt.colorbar()
plt.xlabel('Distance from estuary mouth (km)')
plt.ylabel('Elevation (m MSL)')
plt.title('Difference of settling flux (tidally averaged)')
cbar.ax.set_ylabel(r'$kg/m^2/s$')
#plt.triplot(triang, 'ko-')
#plt.plot(r3d.X[node_thalweg_bottom],  varPlane[r3d.id_z,node_thalweg_bottom],  'ro')
#plt.plot(r3d.X[node_thalweg_surface], varPlane[r3d.id_z,node_thalweg_surface], 'bo')
plt.show()

r3d.slf.close()