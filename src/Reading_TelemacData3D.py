"""
Reading_TelemacData3D

Date:08/08/2019
This module reads the output data from
a Telemac-3D simulation and plots subsequently
some variables (e.g. water-level, flow velocity ...)

Authors: Q.Bi
-------------------------------------------------

Date:21/8/2019
Add functions for interpolation and extrapolation on fixed grid (Q. Bi)

"""

import numpy as np
import math
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import ppmodules.selafin_io_pp as sio
from scipy import interpolate
from scipy.fftpack import fft,fftfreq,fftshift,rfft,irfft,ifft
import gc

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
            # plt.plot(varProfile[n],self.Z[n])
            n = n+1
        # plt.xlabel(self.slf.vars[id_var])
        # plt.ylabel('Elevation (m)')
        # plt.show()
        return varProfile

    def interp_on_fixed_elevations(self, varProfile, Z_bottom, Z_surface, nr_layers):
        Z_new = np.linspace(Z_bottom, Z_surface, nr_layers)
        profile_new = []
        for i in range(len(varProfile)):
            Z = self.Z[i]
            profile = varProfile[i]
            spl = interpolate.make_interp_spline(Z,profile,bc_type='natural')            
            profile_new.append(spl(Z_new))
            # plt.plot(profile,Z,'o-',profile_new,Z_new,'-')
            # plt.show()
        return Z_new, profile_new

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
        plt.show()
        return node_thalweg3D, X_thalweg, Z_thalweg, varPlane

    def cal_U_harmonic_analysis_FFT(self, node_list, start_t, end_t, Z_bottom, Z_surface, nr_layers, time_interval):
        # start reading and processing the data
        M2_thalweg_U  = []
        M4_thalweg_U  = []
        M6_thalweg_U  = []
        M8_thalweg_U  = []
        for iPoint in node_list:
            i = iPoint - (node_list[0] - 1)
            # print('Processing the node', str(i))
            # extract time series along vertical planes
            IPOIN3D, raw_profiles = self.get_raw_profile(iPoint)
            varProfile = self.get_var_profile(raw_profiles, self.id_u, IPOIN3D, start_t, end_t)
            # interpolation and extrapolation
            Z_new, profile_new = self.interp_on_fixed_elevations(varProfile, Z_bottom, Z_surface, nr_layers)
            # reconstruct timeseries and perform FFT
            M2_profile_U  = np.zeros([nr_layers, 3])
            M4_profile_U  = np.zeros([nr_layers, 3])
            M6_profile_U  = np.zeros([nr_layers, 3])
            M8_profile_U  = np.zeros([nr_layers, 3])
            for iLayer in range(nr_layers):
                print('Processing the node', str(i),'on plane',str(iLayer))
                timeseries = np.array([profile_new[j][iLayer] for j in range(len(profile_new))])
                # plt.plot(r3d.time[start_t:end_t+1],timeseries)
                # plt.show()
                # FFT for U
                N = len(timeseries) # number of smaples
                T = time_interval # time intreval (s)
                sp = fft(timeseries)
                freq = 2*np.pi*fftfreq(N,T)
                amp = 2.0/N*np.abs(sp)
                phi = np.angle(sp)*180/np.pi # phase in degree 
                # get the index of M2, M4, M6 and M8
                id_m2 = (np.abs(freq - 1.4056343e-4)).argmin()
                id_m4 = (np.abs(freq - 2*1.4056343e-4)).argmin()
                id_m6 = (np.abs(freq - 3*1.4056343e-4)).argmin()
                id_m8 = (np.abs(freq - 4*1.4056343e-4)).argmin()
                # get the amp, freq and phase of each component
                M2_profile_U[iLayer,:] = [freq[id_m2],amp[id_m2],phi[id_m2]]
                M4_profile_U[iLayer,:] = [freq[id_m4],amp[id_m4],phi[id_m4]]
                M6_profile_U[iLayer,:] = [freq[id_m6],amp[id_m6],phi[id_m6]]
                M8_profile_U[iLayer,:] = [freq[id_m8],amp[id_m8],phi[id_m8]]
            M2_thalweg_U.append(M2_profile_U)
            M4_thalweg_U.append(M4_profile_U)
            M6_thalweg_U.append(M6_profile_U)
            M8_thalweg_U.append(M8_profile_U)
        # get the X coordinates for later plotting figures
        X_new = np.array([self.X[i-1] for i in node_list])/1000 # convert m to km
        # save the processed data
        np.savez('harmonic_analysis_U', X_new=X_new, Z_new=Z_new, M2_thalweg_U=M2_thalweg_U, M4_thalweg_U=M4_thalweg_U, M6_thalweg_U=M6_thalweg_U, M8_thalweg_U=M8_thalweg_U)
        # output of the processed results
        return X_new, Z_new, M2_thalweg_U, M4_thalweg_U

    def plot_U_harmonic_analysis(self, node_list, X_new, Z_new, M2_thalweg_U, M4_thalweg_U):
        # for plot the figures using contourf
        grid_x, grid_z = np.meshgrid(X_new,Z_new)
        # get the amplitude of M2 component
        tempM2 = np.zeros_like(grid_x)
        for i in range(len(node_list)):
            tempM2[:,i] = M2_thalweg_U[i][:,1]
        # get the  amplitude of M4 component
        tempM4 = np.zeros_like(grid_x)
        for i in range(len(node_list)):
            tempM4[:,i] = M4_thalweg_U[i][:,1]
        # making figures
        fig, axs = plt.subplots(2,1,constrained_layout=True)
        cs0 = axs[0].contourf(grid_x, grid_z, tempM2, 100)
        cs1 = axs[1].contourf(grid_x, grid_z, tempM4, 100)
        axs[0].set_title('Horizontal velocity M2 Amplitude(m)')
        axs[1].set_title('Horizontal velocity M4 Amplitude(m)')
        axs[0].set_xlabel('Distance from downstream BC (km)')
        axs[1].set_xlabel('Distance from downstream BC (km)')
        axs[0].set_ylabel('Elevation (m)')
        axs[1].set_ylabel('Elevation (m)')
        fig.colorbar(cs0,ax=axs[0])
        fig.colorbar(cs1,ax=axs[1])
        fig.set_size_inches(10, 5)
        fig.savefig('M2_M4_amp_interpolated_U_at_fixed_elevations.png', dpi=300)
        plt.close()
        # get the phase of M2 component
        tempM2 = np.zeros_like(grid_x)
        for i in range(len(node_list)):
            tempM2[:,i] = M2_thalweg_U[i][:,2]
        # get the phase of M4 component
        tempM4 = np.zeros_like(grid_x)
        for i in range(len(node_list)):
            tempM4[:,i] = M4_thalweg_U[i][:,2]
        # making the figures
        fig, axs = plt.subplots(2,1,constrained_layout=True)
        cs0 = axs[0].contourf(grid_x, grid_z, tempM2, 100)
        cs1 = axs[1].contourf(grid_x, grid_z, tempM4, 100)
        axs[0].set_title('Horizontal velocity M2 phase(degree)')
        axs[1].set_title('Horizontal velocity M4 phase(degree)')
        axs[0].set_xlabel('Distance from downstream BC (km)')
        axs[1].set_xlabel('Distance from downstream BC (km)')
        axs[0].set_ylabel('Elevation (m)')
        axs[1].set_ylabel('Elevation (m)')
        fig.colorbar(cs0,ax=axs[0])
        fig.colorbar(cs1,ax=axs[1])
        fig.set_size_inches(10, 5)
        fig.savefig('M2_M4_phs_interpolated_U_at_fixed_elevations.png', dpi=300)
        plt.close()
        return

#--------------------------------------#--------------------------------------#--------------------------------------
# test 1 - computing sediment and settling fluxes
slf_file = 'D:\\QILONG_BI\\Research\\MCPBE_flocculation_model\\idealized_Scheldt\\r3D_idealized_Scheldt_2CPBE.slf'
slf_file = 'D:\\QILONG_BI\\Research\\MCPBE_flocculation_model\\idealized_Scheldt\\r3D_idealized_Scheldt.slf'
r3d = Reading_TelemacData3D(slf_file)

IPOIN2D = 6539
start_t = 200
end_t = 260

IPOIN3D, raw_profiles = r3d.get_raw_profile(IPOIN2D)
varProfile = r3d.get_var_profile(raw_profiles, 10, IPOIN3D, start_t, end_t)

#--------------------------------------
timestep = 252
for timestep in range(200,260):
    node_thalweg3D, X_thalweg, Z_thalweg, varPlane = r3d.get_2DV_slice_thalweg(timestep,10,100)

#--------------------------------------
# settling flux
timestep = 252
r3d.varNames
r3d.slf.readVariables(timestep-1)
varPlane = np.array(r3d.slf.getVarValues())
# find thalweg points
node_thalweg3D = [i for i in range(len(r3d.Y)) if r3d.Y[i]==0]
X_thalweg = np.array([r3d.X[i] for i in range(len(r3d.X)) if r3d.Y[i]==0])
Z_thalweg = varPlane[r3d.id_z,node_thalweg3D]

C = (varPlane[8,node_thalweg3D]+varPlane[10,node_thalweg3D])*varPlane[6,node_thalweg3D]
C = (varPlane[8,node_thalweg3D])*1.1006341463414635E-003

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
plt.tricontourf(triang, C, levels=200, cmap='bwr')
cbar = plt.colorbar()
cbar.ax.set_ylabel('lalala')
#plt.triplot(triang, 'ko-')
#plt.plot(r3d.X[node_thalweg_bottom],  varPlane[r3d.id_z,node_thalweg_bottom],  'ro')
#plt.plot(r3d.X[node_thalweg_surface], varPlane[r3d.id_z,node_thalweg_surface], 'bo')
plt.show()

#--------------------------------------#--------------------------------------#--------------------------------------
# test 2 - computing U at fixed elevations
slf_file = r'C:\Users\saaad264\Research\19_025_Telemac-iFlow\Telemac_model\rectangular_mesh\am2_1.0_am4_0.05_h_20.0\r3D_rectangular_mesh.slf'
r3d = Reading_TelemacData3D(slf_file)
# thalweg points in 2D
ip_start = 2563
ip_end = 3843
node_list = range(ip_start, ip_end + 1)
# beginning and end time frames 
start_t = 745
end_t = 5662
time_interval = 600.0
# interpolate velocity on fixed elevations
Z_bottom = -10
Z_surface = 0
nr_layers = 11 # total number of layers, including bottom and surface
# perform the computation
X_new, Z_new, M2_thalweg_U, M4_thalweg_U = r3d.cal_U_harmonic_analysis_FFT(node_list, start_t, end_t, Z_bottom, Z_surface, nr_layers, time_interval)
r3d.plot_U_harmonic_analysis(node_list, X_new, Z_new, M2_thalweg_U, M4_thalweg_U)
r3d.slf.close() # close files



