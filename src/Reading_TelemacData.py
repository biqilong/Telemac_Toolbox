"""
Reading_TelemacData

Date:9/7/2019
This module reads the output data from
a Telemac simulation and plots subsequently
some variables (e.g. water-level, flow velocity ...)

Authors: S.J. Kaptein, Q.Bi
-------------------------------------------------

Date:23/7/2019
Rearrange the functions (Q. Bi)

Date:13/8/2019
Add function for computing tidal components using FFT (Q. Bi)

"""

import numpy as np
import math
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import ppmodules.selafin_io_pp as sio
import ttide as tt
import os
import csv
from scipy.fftpack import fft,fftfreq,fftshift,rfft,irfft,ifft


class Reading_TelemacData:
    # Variables

    # Methods
    def __init__(self, input):
        self.input = input

        return

    def run(self,post_input):

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
        slf, varNames, id_u, id_v, id_wl, id_ustar, time, XY = self.TelemacReadHeader(slf_file_2D)

        # timestep for extracting 2D results at certain time step
        timestep = 2000
        # node_nr for extracting time series at certain node
        node_nr = 2563  # telemac node nr, in python should be node_nr-1
        # extract 2D field of a variable
        grid_x, grid_y, varField, grid_Var = self.TelemacReadData2D(slf, XY, timestep, id_ustar)
        self.TelemacPlotFlowfields(fig_dir, timestep, grid_x, grid_y, grid_Var, varNames, id_ustar)
        self.TelemacPlotFlowUV(fig_dir, timestep, grid_x, grid_y, id_u, id_v, XY, varField)

        # extract time series of all the variables at one node
        timeseries = self.TelemacReadDataTS(slf, node_nr-1)
        self.PlotTelemacTimeseries(fig_dir, time, timeseries, node_nr, id_wl, id_u, id_ustar)

        # compute M2 and M4 using t-tide
        # M2_thalweg_WL, M4_thalweg_WL, M2_thalweg_U, M4_thalweg_U = self.CalculateM2M4(slf, XY, id_wl, id_u, ip_start, ip_end, tf_start, tf_end, pp_dir)
        # self.PlotTelemacM2M4(fig_dir, ip_start, ip_end, XY, M2_thalweg_WL, M4_thalweg_WL, M2_thalweg_U, M4_thalweg_U)
        M2_thalweg_WL, M4_thalweg_WL, M6_thalweg_WL, M8_thalweg_WL, M2_thalweg_U, M4_thalweg_U, M6_thalweg_U, M8_thalweg_U = self.CalculateM2M4_FFT(slf, XY, id_wl, id_u, ip_start, ip_end, tf_start, tf_end, time_interval, pp_dir)


        # data container
        d = dict()
        d['telemacData'] = {}
        d['telemacData']['x'] = XY[ip_start-1:ip_end,0]
        d['telemacData']['zeta0'] = M2_thalweg_WL
        d['telemacData']['zeta1'] = M4_thalweg_WL
        d['telemacData']['u0'] = M2_thalweg_U
        d['telemacData']['u1'] = M4_thalweg_U

        ts_downstream_bc = self.TelemacReadDataTS(slf, ip_start-1)
        d['telemacData']['timeseries'] = {}
        d['telemacData']['timeseries']['time'] = time
        d['telemacData']['timeseries']['u'] = ts_downstream_bc[:, id_u]
        d['telemacData']['timeseries']['wl'] = ts_downstream_bc[:, id_wl]

        return d

    def TelemacReadHeader(self, slf_file_2D):
        """This function reads the header of Selafin file"""
        # read the file header
        slf = sio.ppSELAFIN(slf_file_2D)
        slf.readHeader()

        # get the index of the variables
        varNames = slf.getVarNames()
        for i in range(0,slf.NBV1): # from 0 to slf.NBV1-1
            if varNames[i]=='VELOCITY U      ':
                id_u = i
            if varNames[i]=='VELOCITY V      ':
                id_v = i
            if varNames[i]=='FREE SURFACE    ':
                id_wl = i
            if varNames[i]=='FRICTION VELOCI ':
                id_ustar = i

        # read time stamps
        slf.readTimes()
        time = slf.getTimes()
        print('Total number of timesteps: ' + str(len(time)) + '\n')

        # read mesh coordinates
        X = slf.x
        Y = slf.y
        XY = np.concatenate((np.reshape(X,(slf.NPOIN,1)),np.reshape(Y,(slf.NPOIN,1))),axis=1)

        return(slf, varNames, id_u, id_v, id_wl, id_ustar, time, XY)

    def TelemacReadData2D(self, slf, XY, timestep, id_Var):
        """This function reads the 2D field from the Selafin file"""

        # read variables at desired timestep
        slf.readVariables(timestep)
        varField = slf.getVarValues()  # shape (NBV1,NPOIN)

        # map the variable field to 2D grid
        grid_x, grid_y = np.mgrid[0:160000 + 1000:1000, -250:250 + 50:50]  # size of the domain
        grid_Var = griddata(XY, varField[id_Var, :], (grid_x, grid_y), method='linear')

        return(grid_x, grid_y, varField, grid_Var)

    def TelemacReadDataTS(self, slf, node_nr):
        """This function reads the time series from the Selafin file"""

        # read time series at the desired node
        slf.readVariablesAtNode(node_nr)
        timeseries = slf.getVarValuesAtNode()

        return(timeseries)

    def TelemacPlotFlowUV(self, fig_dir, timestep, grid_x, grid_y, id_u, id_v, XY, varField):
        """This function generates plot of UV vectors"""

        grid_U = griddata(XY, varField[id_u, :], (grid_x, grid_y), method='linear')
        grid_V = griddata(XY, varField[id_v, :], (grid_x, grid_y), method='linear')

        fig, axs = plt.subplots()
        axs.quiver(grid_x, grid_y, grid_U, grid_V)
        #plt.show()
        plt.savefig(fig_dir+"velocity_field_at_timestep"+str(timestep)+".png", dpi=300)
        plt.close()

        return

    def TelemacPlotFlowfields(self, fig_dir, timestep, grid_x, grid_y, grid_Var, varNames, id_Var):
        """This function generates plot of flow field related variables"""
        
        fig, axs = plt.subplots()
        ### comment SKN to QBI: I had to replace 'levels=100' by '100', otherwise I had an error
        axs.contourf(grid_x, grid_y, grid_Var, 100)
        #plt.show()
        plt.savefig(fig_dir+varNames[id_Var].rstrip()+"_at_timestep"+str(timestep)+".png", dpi=300)
        plt.close()

        # plot variable along thalweg
        fig, axs = plt.subplots()
        id_thalweg = math.floor(np.shape(grid_x)[1] / 2)
        id_thalweg = int(id_thalweg)
        ### comment SKN to QBI: For some reasons id_thalweg is not an integer, so python complains, could it be because np.shape is returning?
        axs.plot(grid_x[:, id_thalweg], grid_Var[:, id_thalweg])
        #plt.show()
        plt.savefig(fig_dir+varNames[id_Var].rstrip()+"_at_timestep"+str(timestep)+"_along_thalweg.png", dpi=300)
        plt.close()

        return

    def PlotTelemacTimeseries(self, fig_dir, time, timeseries, node_nr, id_wl, id_u, id_ustar):
        """This function generates a plot of the timeseries at a specific location"""

        # flow_dir = np.divide(timeseries[:,id_u],np.absolute(timeseries[:,id_u]))

        # Plot free surface, velocity and bed shear stress
        fig, axs = plt.subplots(3, 1)
        axs[0].plot(time, timeseries[:, id_wl])
        axs[0].set_xlim(0, time[-1])
        axs[0].set_xlabel('time')
        axs[0].set_ylabel('Free surface (m)')
        axs[0].grid(True)
        axs[1].plot(time, timeseries[:, id_u])
        axs[1].set_xlim(0, time[-1])
        axs[1].set_xlabel('time')
        axs[1].set_ylabel('Velocity U (m/s)')
        axs[1].grid(True)
        axs[2].plot(time, np.square(timeseries[:, id_ustar]) * 1000.0)
        axs[2].set_xlim(0, time[-1])
        axs[2].set_xlabel('time')
        axs[2].set_ylabel('Bed shear stress (Pa)')
        axs[2].grid(True)
        fig.tight_layout()
        #plt.show()
        plt.savefig(fig_dir+"timeseries_of_node_"+str(node_nr)+".png", dpi=300)
        plt.close()

        return

    def CalculateM2M4(self, slf, XY, id_wl, id_u, ip_start, ip_end, tf_start, tf_end, pp_dir):
        # calculate M2 and M4 along the channel using ttide_py

        M2outfileName = pp_dir+'M2_along_thalweg.out'
        M4outfileName = pp_dir+'M4_along_thalweg.out'

        # the computation of the M2 and M4 components of the Telemac data might be expensive.
        # as a result, it is first checked if the M2 and M4 components are not already computed and saved to an outputfile
        # if this is the case, the M2 and M4 components are obtained from the output file instead of being computed
        if (os.path.isfile(M2outfileName)):
            M2outputfile = open(M2outfileName, 'r')
            M2lineNb = M2outputfile.readline().split('xlength=')[1]
            M2lineNb = int(M2lineNb.replace(']', ''))
            M2outputfile.close()
        if (os.path.isfile(M4outfileName)):
            M4outputfile = open(M4outfileName, 'r')
            M4lineNb = M4outputfile.readline().split('xlength=')[1]
            M4lineNb = int(M4lineNb.replace(']', ''))
            M4outputfile.close()

        if (os.path.isfile(M2outfileName)) and (M2lineNb == ip_end-ip_start+1) and (os.path.isfile(M4outfileName)) and (M4lineNb == ip_end-ip_start+1):
            print ('all outputfiles exist and have the correct number of lines')

            # here there might be some issues between python 3 and python 2
            with open(M2outfileName, 'r') as M2outputfile:
                M2outputfile.readline()
                dataMatrix = list(csv.reader(M2outputfile))
                dataMatrix = np.array(dataMatrix)
                dataMatrix = dataMatrix.astype(np.float)

            #XY=dataMatrix[:, 0:2]
            M2_thalweg_WL = dataMatrix[:, 2:6]
            M2_thalweg_U  = dataMatrix[:, 6:10]

            with open(M4outfileName, 'r') as M4outputfile:
                M4outputfile.readline()
                dataMatrix = list(csv.reader(M4outputfile))
                dataMatrix = np.array(dataMatrix)
                dataMatrix = dataMatrix.astype(np.float)

            #XY=dataMatrix[:, 0:2]
            M4_thalweg_WL = dataMatrix[:, 2:6]
            M4_thalweg_U  = dataMatrix[:, 6:10]

#           M2outputfile.close()
#           M4outputfile.close()

        elif not (os.path.isfile(M2outfileName) or os.path.isfile(M4outfileName)):
            print('not all outputfile exist or have the correct number of lines, M2 and M4 are computed')

            M2_thalweg_WL = np.zeros([ip_end - ip_start + 1, 4])
            M4_thalweg_WL = np.zeros([ip_end - ip_start + 1, 4])
            M2_thalweg_U  = np.zeros([ip_end - ip_start + 1, 4])
            M4_thalweg_U  = np.zeros([ip_end - ip_start + 1, 4])

            # python starts from ip_start-1 to ip_end-1
            for iPoint in range(ip_start - 1, ip_end):
                i = iPoint - (ip_start - 1)
                #i = iPoint - (ip_start - 1)+1
                print('Processing the node', str(i))
                # read time series at the desired node
                slf.readVariablesAtNode(iPoint)
                timeseries = slf.getVarValuesAtNode()
                # harmonic analysis using t-tide with results every 10 min
                tideconout_WL = tt.t_tide(timeseries[:,id_wl][tf_start:tf_end],dt=1.0/6.0,
                                          constitnames=['M2', 'M4', 'M6', 'M8'],out_style=None) # id_wl for water level
                tideconout_U  = tt.t_tide(timeseries[:, id_u][tf_start:tf_end],dt=1.0/6.0,
                                          constitnames=['M2', 'M4', 'M6', 'M8'],out_style=None) # id_u for depth-averaged horizontal velocity

                # get M2 and M4 from WL
                id_M2_WL = list(tideconout_WL["nameu"]).index(b'M2  ')  # look for M2 constituent
                id_M4_WL = list(tideconout_WL["nameu"]).index(b'M4  ')  # look for M4 constituent
                M2_thalweg_WL[i, :] = tideconout_WL["tidecon"][id_M2_WL]  # [amp err_amp phase err_phase]
                M4_thalweg_WL[i, :] = tideconout_WL["tidecon"][id_M4_WL]  # [amp err_amp phase err_phase]
                # get M2 and M4 from U
                id_M2_U = list(tideconout_U["nameu"]).index(b'M2  ')  # look for M2 constituent
                id_M4_U = list(tideconout_U["nameu"]).index(b'M4  ')  # look for M4 constituent
                M2_thalweg_U[i, :] = tideconout_U["tidecon"][id_M2_U]  # [amp err_amp phase err_phase]
                M4_thalweg_U[i, :] = tideconout_U["tidecon"][id_M4_U]  # [amp err_amp phase err_phase]

            # save the data to text files
            M2_out = np.concatenate((XY[ip_start - 1:ip_end, :], M2_thalweg_WL, M2_thalweg_U), axis=1)
            M4_out = np.concatenate((XY[ip_start - 1:ip_end, :], M4_thalweg_WL, M4_thalweg_U), axis=1)

            if not os.path.exists(pp_dir):
                os.mkdir(pp_dir)
            
            np.savetxt(M2outfileName, M2_out, delimiter=',',
                       header='[X,Y,WL_amp,WL_err_amp,WL_phase,WL_err_phase,U_amp,U_err_amp,U_phase,U_err_phase,xlength='+str(ip_end-ip_start+1)+']')
            np.savetxt(M4outfileName, M4_out, delimiter=',',
                       header='[X,Y,WL_amp,WL_err_amp,WL_phase,WL_err_phase,U_amp,U_err_amp,U_phase,U_err_phase,xlength='+str(ip_end-ip_start+1)+']')

        return(M2_thalweg_WL, M4_thalweg_WL, M2_thalweg_U, M4_thalweg_U)


    def CalculateM2M4_FFT(self, slf, XY, id_wl, id_u, ip_start, ip_end, tf_start, tf_end, time_interval, pp_dir):
        # calculate M2 and M4 along the channel using ttide_py

        M2_thalweg_WL = np.zeros([ip_end - ip_start + 1, 3])
        M4_thalweg_WL = np.zeros([ip_end - ip_start + 1, 3])
        M6_thalweg_WL = np.zeros([ip_end - ip_start + 1, 3])
        M8_thalweg_WL = np.zeros([ip_end - ip_start + 1, 3])
        M2_thalweg_U  = np.zeros([ip_end - ip_start + 1, 3])
        M4_thalweg_U  = np.zeros([ip_end - ip_start + 1, 3])
        M6_thalweg_U  = np.zeros([ip_end - ip_start + 1, 3])
        M8_thalweg_U  = np.zeros([ip_end - ip_start + 1, 3])

        M2outfileName = pp_dir+'M2_along_thalweg_FFT.out'
        M4outfileName = pp_dir+'M4_along_thalweg_FFT.out'
        M6outfileName = pp_dir+'M6_along_thalweg_FFT.out'
        M8outfileName = pp_dir+'M8_along_thalweg_FFT.out'

        if (os.path.isfile(M2outfileName)):
            M2outputfile = open(M2outfileName, 'r')
            M2lineNb = M2outputfile.readline().split('xlength=')[1]
            M2lineNb = int(M2lineNb.replace(']', ''))
            M2outputfile.close()
        if (os.path.isfile(M4outfileName)):
            M4outputfile = open(M4outfileName, 'r')
            M4lineNb = M4outputfile.readline().split('xlength=')[1]
            M4lineNb = int(M4lineNb.replace(']', ''))
            M4outputfile.close()
        if (os.path.isfile(M6outfileName)):
            M6outputfile = open(M6outfileName, 'r')
            M6lineNb = M6outputfile.readline().split('xlength=')[1]
            M6lineNb = int(M6lineNb.replace(']', ''))
            M2outputfile.close()
        if (os.path.isfile(M8outfileName)):
            M8outputfile = open(M8outfileName, 'r')
            M8lineNb = M8outputfile.readline().split('xlength=')[1]
            M8lineNb = int(M8lineNb.replace(']', ''))
            M8outputfile.close()

        if os.path.isfile(M2outfileName) and os.path.isfile(M4outfileName) and os.path.isfile(M6outfileName) and os.path.isfile(M8outfileName) and (M2lineNb == ip_end-ip_start+1) and (M4lineNb == ip_end-ip_start+1) and (M6lineNb == ip_end-ip_start+1) and (M8lineNb == ip_end-ip_start+1):
            print('all outputfiles exist and have the correct number of lines')
        else:
            # python starts from ip_start-1 to ip_end-1
            for iPoint in range(ip_start - 1, ip_end):
                i = iPoint - (ip_start - 1)
                #i = iPoint - (ip_start - 1)+1
                print('Processing the node', str(i))
                # read time series at the desired node
                slf.readVariablesAtNode(iPoint)
                timeseries = slf.getVarValuesAtNode()
                # harmonic analysis using FFT with results every 10 min
                # FFT for WL
                N = len(timeseries[:,id_wl][tf_start:tf_end]) # number of smaples
                T = time_interval # time intreval (s)
                sp = fft(timeseries[:,id_wl][tf_start:tf_end])
                freq = 2*np.pi*fftfreq(N,time_interval)
                amp = 2.0/N * np.abs(sp)
                phi = np.angle(sp)*180/np.pi # phase in degree 
                # get the index of M2, M4, M6 and M8
                id_m2 = (np.abs(freq - 1.4056343e-4)).argmin()
                id_m4 = (np.abs(freq - 2*1.4056343e-4)).argmin()
                id_m6 = (np.abs(freq - 3*1.4056343e-4)).argmin()
                id_m8 = (np.abs(freq - 4*1.4056343e-4)).argmin()
                # get the amp, freq and phase of each component
                M2_thalweg_WL[i,:] = [freq[id_m2],amp[id_m2],phi[id_m2]]
                M4_thalweg_WL[i,:] = [freq[id_m4],amp[id_m4],phi[id_m4]]
                M6_thalweg_WL[i,:] = [freq[id_m6],amp[id_m6],phi[id_m6]]
                M8_thalweg_WL[i,:] = [freq[id_m8],amp[id_m8],phi[id_m8]]
                # FFT for U
                N = len(timeseries[:,id_u][tf_start:tf_end]) # number of smaples
                T = time_interval # time intreval (s)
                sp = fft(timeseries[:,id_u][tf_start:tf_end])
                freq = 2*np.pi*fftfreq(N,T)
                amp = 2.0/N * np.abs(sp)
                phi = np.angle(sp)*180/np.pi # phase in degree 
                # get the index of M2, M4, M6 and M8
                id_m2 = (np.abs(freq - 1.4056343e-4)).argmin()
                id_m4 = (np.abs(freq - 2*1.4056343e-4)).argmin()
                id_m6 = (np.abs(freq - 3*1.4056343e-4)).argmin()
                id_m8 = (np.abs(freq - 4*1.4056343e-4)).argmin()
                # get the amp, freq and phase of each component
                M2_thalweg_U[i,:] = [freq[id_m2],amp[id_m2],phi[id_m2]]
                M4_thalweg_U[i,:] = [freq[id_m4],amp[id_m4],phi[id_m4]]
                M6_thalweg_U[i,:] = [freq[id_m6],amp[id_m6],phi[id_m6]]
                M8_thalweg_U[i,:] = [freq[id_m8],amp[id_m8],phi[id_m8]]
            # save the data to text files
            M2_out = np.concatenate((XY[ip_start - 1:ip_end, :], M2_thalweg_WL, M2_thalweg_U), axis=1)
            M4_out = np.concatenate((XY[ip_start - 1:ip_end, :], M4_thalweg_WL, M4_thalweg_U), axis=1)
            M6_out = np.concatenate((XY[ip_start - 1:ip_end, :], M6_thalweg_WL, M6_thalweg_U), axis=1)
            M8_out = np.concatenate((XY[ip_start - 1:ip_end, :], M8_thalweg_WL, M8_thalweg_U), axis=1)

            if not os.path.exists(pp_dir):
                os.mkdir(pp_dir)
            
            np.savetxt(M2outfileName, M2_out, delimiter=',',
                        header='[X,Y,WL_freq,WL_amp,WL_phase,U_freq,U_amp,U_phase,xlength='+str(ip_end-ip_start+1)+']')
            np.savetxt(M4outfileName, M4_out, delimiter=',',
                        header='[X,Y,WL_freq,WL_amp,WL_phase,U_freq,U_amp,U_phase,xlength='+str(ip_end-ip_start+1)+']')
            np.savetxt(M6outfileName, M6_out, delimiter=',',
                        header='[X,Y,WL_freq,WL_amp,WL_phase,U_freq,U_amp,U_phase,xlength='+str(ip_end-ip_start+1)+']')
            np.savetxt(M8outfileName, M8_out, delimiter=',',
                        header='[X,Y,WL_freq,WL_amp,WL_phase,U_freq,U_amp,U_phase,xlength='+str(ip_end-ip_start+1)+']')

        return(M2_thalweg_WL, M4_thalweg_WL, M6_thalweg_WL, M8_thalweg_WL, M2_thalweg_U, M4_thalweg_U, M6_thalweg_U, M8_thalweg_U)


    def PlotTelemacM2M4(self, fig_dir, ip_start, ip_end, XY, M2_thalweg_WL, M4_thalweg_WL, M2_thalweg_U, M4_thalweg_U):

        # todo read the data from the post processed file

        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)

        # Plot M2 and M4 along thalweg
        fig, axs = plt.subplots(2, 2)
        axs[0][0].plot(XY[ip_start-1:ip_end,0], M2_thalweg_WL[:, 0])
        axs[0][0].set_xlim(XY[ip_start-1:ip_end,0][0], XY[ip_start-1:ip_end,0][-1])
        axs[0][0].set_xlabel('Distance (m)')
        axs[0][0].set_ylabel('WL M2 amplitude (m)')
        axs[0][0].grid(True)
        axs[0][1].plot(XY[ip_start-1:ip_end,0], M2_thalweg_WL[:, 2])
        axs[0][1].set_xlim(XY[ip_start-1:ip_end,0][0], XY[ip_start-1:ip_end,0][-1])
        axs[0][1].set_xlabel('Distance (m)')
        axs[0][1].set_ylabel('WL M2 phase (m)')
        axs[0][1].grid(True)
        axs[1][0].plot(XY[ip_start-1:ip_end,0], M4_thalweg_WL[:, 0])
        axs[1][0].set_xlim(XY[ip_start-1:ip_end,0][0], XY[ip_start-1:ip_end,0][-1])
        axs[1][0].set_xlabel('Distance (m)')
        axs[1][0].set_ylabel('WL M4 amplitude (m)')
        axs[1][0].grid(True)
        axs[1][1].plot(XY[ip_start-1:ip_end,0], M4_thalweg_WL[:, 2])
        axs[1][1].set_xlim(XY[ip_start-1:ip_end,0][0], XY[ip_start-1:ip_end,0][-1])
        axs[1][1].set_xlabel('Distance (m)')
        axs[1][1].set_ylabel('WL M4 phase (m)')
        axs[1][1].grid(True)
        fig.tight_layout()
        # plt.show()
        plt.savefig(fig_dir+"water_level_M2_M4_along_thalweg.png", dpi=300)
        plt.close()
        
        fig, axs = plt.subplots(2, 2)
        axs[0][0].plot(XY[ip_start-1:ip_end,0], M2_thalweg_U[:, 0])
        axs[0][0].set_xlim(XY[ip_start-1:ip_end,0][0], XY[ip_start-1:ip_end,0][-1])
        axs[0][0].set_xlabel('Distance (m)')
        axs[0][0].set_ylabel('U M2 amplitude (m)')
        axs[0][0].grid(True)
        axs[0][1].plot(XY[ip_start-1:ip_end,0], M2_thalweg_U[:, 2])
        axs[0][1].set_xlim(XY[ip_start-1:ip_end,0][0], XY[ip_start-1:ip_end,0][-1])
        axs[0][1].set_xlabel('Distance (m)')
        axs[0][1].set_ylabel('U M2 phase (m)')
        axs[0][1].grid(True)
        axs[1][0].plot(XY[ip_start-1:ip_end,0], M4_thalweg_U[:, 0])
        axs[1][0].set_xlim(XY[ip_start-1:ip_end,0][0], XY[ip_start-1:ip_end,0][-1])
        axs[1][0].set_xlabel('Distance (m)')
        axs[1][0].set_ylabel('U M4 amplitude (m)')
        axs[1][0].grid(True)
        axs[1][1].plot(XY[ip_start-1:ip_end,0], M4_thalweg_U[:, 2])
        axs[1][1].set_xlim(XY[ip_start-1:ip_end,0][0], XY[ip_start-1:ip_end,0][-1])
        axs[1][1].set_xlabel('Distance (m)')
        axs[1][1].set_ylabel('U M4 phase (m)')
        axs[1][1].grid(True)
        fig.tight_layout()
        # plt.show()
        plt.savefig(fig_dir+"velocity_U_M2_M4_along_thalweg.png", dpi=300)
        plt.close()

        # todo use data container

        return

