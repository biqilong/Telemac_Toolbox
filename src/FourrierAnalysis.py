"""
Extract the leading frequency and its harmonics of a signal

Date: 08-11-19
Authors: S.J. Kaptein

contains the functions:

    tidalSignal(time, data, jx, omega, modeNb)
    interpExactPeriod(x, y, period)
    computeSpectrum(x, y, omega)
    extractMfrequencies(freq, A, phi, omega, modeNb)
    reconstructSeries(amplitude, phase, time, omega, maxMode)
    constructVariables(amplitude, phase, order)
    definePeriods(timeSeries, period)

"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.fftpack import fft,fftfreq,fftshift,rfft,irfft,ifft
import time as timemod
import matplotlib.pyplot as plt
# from plotting import FastPlot as fplt
# from plotting import PlotSettings as pltset
import sys

#todo: adapt interpExactPeriod so that it can be used by TelemacInputOutput

def tidalSignals(time, data, jx, omega, modeNbList, order):
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    #
    # this function mainly calls the other functions of this same module
    # it computes and saves into a dictionary   the original timeseries             : TS_timeOriginal, TS_varOriginal
    #                                           the interpolated timeseries         : TS_timeInterp, TS_varInterp
    #                                           the reconstructed timeseries        : TS_reconstructed04, TS_reconstructedMax
    #                                           the harmonic analysis quantities    : HA_frequency, HA_amplitude, HA_phase
    #                                           mode quantities                     : modes_amplitude, modes_phase
    #                                           reconstructed variables (iFlow type): variable0, variable1
    #
    # ARGUMENTS
    #
    # time          : 1d vectors[time]
    # data          : 2d matrix with[time, jx]
    # jx, modeNb    : integer
    # omega         : float
    #
    # RETURNS
    #
    # dictionary containing     time                    : 1d matrix[time]
    #                           data.transpose()        : 2d matrix[jx, time]
    #                           timeExactPeriod         : 1d matrix[N]          (time series interpolated on an exact period)
    #                           varExactPeriod          : 2d matrix[jx, N]      (time series interpolated on an exact period)
    #                           rs04                    : 2d matrix[jx, N]
    #                           rsMax                   : 2d matrix[jx, N]
    #                           f, A, phi               : 2d matrices[jx, N]    (frequency, angle and phase of fourier analysis)
    #                           amplitude, phase        : 2d matrices[jx, frequency]
    #                           variable0, variable1    : 2d matrix[jx, order]
    #
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################


    N = 90


    period = 2. * np.arccos(-1.) / omega

    f = np.zeros([jx, N])
    A = np.zeros([jx, N])
    phi = np.zeros([jx, N])
    amplitude = np.zeros([jx, max(modeNbList)])
    phase = np.zeros([jx, max(modeNbList)])
    varExactPeriod = np.zeros([jx, N])
    rs_int = np.zeros([N, len(modeNbList)])
    rs = np.zeros([jx, N, len(modeNbList)])


    variable0 = np.zeros([jx, order], dtype=complex)
    variable1 = np.zeros([jx, order], dtype=complex)


    for gridPoint in range(0, jx):


        # interpolate on 90 points describing exactly one period
        timeExactPeriod, varExactPeriod_intm = interpExactPeriod(time, data[:, gridPoint], period, N)
        varExactPeriod[gridPoint, :] = varExactPeriod_intm



        # compute energy spectrum
        f_intm, A_intm, phi_intm = computeSpectrum(timeExactPeriod, varExactPeriod_intm, omega)
        f[gridPoint, :] = f_intm
        A[gridPoint, :] = A_intm
        phi[gridPoint, :] = phi_intm


        # extract the M0, M2, M4, M6 and M8 frequencies
        amplitude_intm, phase_intm = extractMfrequencies(f_intm, A_intm, phi_intm, omega, max(modeNbList))
        amplitude[gridPoint, :] = amplitude_intm
        phase[gridPoint, :] = phase_intm

        # reconstruct timeseries
        for counter, modeNb in enumerate(modeNbList):
            rs_int[:, counter] = reconstructSeries(amplitude_intm, phase_intm, timeExactPeriod, omega, modeNb)
            rs[gridPoint, :, counter] = rs_int[:, counter]




        # construct ordered variables
        variable0_int, variable1_int = constructVariables(amplitude_intm, phase_intm, order)
        variable0[gridPoint, :] = variable0_int
        variable1[gridPoint, :] = variable1_int






    #comment: store output in dictionary
    d = {}
    d["TS_timeOriginal"] = time
    d["TS_varOriginal"] = data.transpose()
    d["TS_timeInterp"] = timeExactPeriod
    d["TS_varInterp"] = varExactPeriod
    d["TS_reconstructed"] = {}
    for counter, modeNb in enumerate(modeNbList):
        d["TS_reconstructed"][str(modeNb)] = rs[:, :, counter]
        # d["TS_reconstructed"][modeNb] = rs[:, :, counter]
    d["HA_frequency"] = f
    d["HA_amplitude"] = A
    d["HA_phase"] = phi
    d["modes_amplitude"] = amplitude
    d["modes_phase"] = phase
    d["variable0"] = variable0
    d["variable1"] = variable1





    return(d)

def interpExactPeriod(x, y, period, N):
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    #
    # this function interpolates a time series on a new grid with an exact integer number of points per period
    # ARGUMENTS
    #
    # x, y      : 1d vectors of same size
    # period    : float
    # N         : integer (an optimum value for N appeared to be 90, according to Y.M. Dijkstra, personal communication)
    #
    # RETURNS
    #
    # x_interp, y_interp    : 1d vectors of size (N)
    #
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    # find start of a period
    periodStart = round(x[0] / period) * period

    # generate data point vector upon which to interpolate
    x_interp = np.asarray([periodStart+float(count) / float(N) * period for count in
                                  range(0, int((x[-1]-x[0])/period) * N)])

    # interpolate on the new data points
    interpfunction = interp1d(x, y, kind='cubic')
    y_interp = interpfunction(x_interp)

    return(x_interp, y_interp)

def computeSpectrum(x, y, omega):
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    #
    # this function computes the energy spectrum of y based on the fundamental frequency omega
    #
    # ARGUMENTS
    #
    # x, y      : 1d vectors of same size
    # omega    : float
    #
    # RETURNS
    #
    # freq, A, phi  : 1d lists
    #
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    N=len(x)

    period = 2. * np.arccos(-1.) / omega
    freq = 2 * np.pi * fftfreq(N, 1. / float(N) * period)
    sp = fft(y)



    A = 2.0 / N * np.abs(sp)
    phi = np.angle(sp) # phase in rad

    return(freq, A, phi)

def extractMfrequencies(freq, A, phi, omega, modeNb):
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    #
    # this function extracts the amplitude and the phase of the leading modes based on an energy spectrum
    #
    # ARGUMENTS
    #
    # freq, A, phi  : 1d vectors
    # omega     : float
    # modeNb    : integer
    #
    # RETURNS
    #
    # amplitude, phase  : 1d lists[mode]
    #
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    # get the index of M2, M4, M6 and M8
    id = [ np.abs(freq - mode * omega).argmin() for mode in range(0, modeNb)]


    # get the amp, freq and phase of each component
    frequency = [freq[id[mod]] for mod in range(0, modeNb)]
    amplitude = [0.5*A[id[mod]] if mod== 0 else A[id[mod]] for mod in range(0, modeNb)]
    phase = [-phi[id[mod]] for mod in range(0, modeNb)]

    return(amplitude, phase)

def reconstructSeries(amplitude, phase, time, omega, maxMode):
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    #
    # this function reconstructs the timeseries based on the harmonic decomposition
    #
    # ARGUMENTS
    #
    # amplitude, phase  : 1d matrices[mode]
    # time              : 1d matrix[time]
    # omega             : float
    # maxMode           : integer
    #
    # RETURNS
    #
    # reconstructedSignal   : 1d matrix[jx]
    #
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    # reconstruct the time series signal based on 4 moon tidal components
    reconstructedSignal = amplitude[0] * np.cos(0. * omega * time - phase[0])
    for mode in range(1, maxMode):
        reconstructedSignal = reconstructedSignal + amplitude[mode] * np.cos(float(mode) * omega * time - phase[mode])

    return(reconstructedSignal)

def constructVariables(amplitude, phase, order):
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    #
    # this function constructs the first and second order variable, using the iFlow standard
    #
    # ARGUMENTS
    #
    # amplitude, phase  : 1d vectors[mode]
    # order             : integer
    #
    # RETURNS
    #
    # variable0, variable1 : 1d vector[order]
    #
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################


    variable0=np.zeros(order, dtype=complex)
    variable1=np.zeros(order, dtype=complex)

    variable0[1] = amplitude[1] * np.exp(-1j * phase[1])
    variable1[0] = amplitude[0] * np.exp(-1j * phase[0])
    variable1[2] = amplitude[2] * np.exp(-1j * phase[2])

    return(variable0, variable1)

def definePeriods(timeSeries, period):
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    #
    # this function determines the index at which starts the last and second last period
    #
    # ARGUMENTS
    #
    # timeseries    : 1d vector
    # period        : float
    #
    # RETURNS
    #
    # iPeriod01, iPeriod02, iPeriod03   : integers (indices of the start of each period)
    #
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    st_definePeriods = timemod.time()

    # determine time series of the two last periods
    secondLastT = int(timeSeries[-1] / period) - 2
    timeBegin = float(secondLastT) * period
    timeInterm = float(secondLastT + 1) * period
    timeEnd = float(secondLastT + 2) * period

    frameTimeP = timeSeries[0]
    counter = 0
    for frameTime in timeSeries:
        if (frameTimeP < timeBegin and frameTime > timeBegin):
            iPeriod01 = counter - 1
        if (frameTimeP < timeInterm and frameTime > timeInterm):
            iPeriod02 = counter
        if (frameTimeP < timeEnd and frameTime > timeEnd):
            iPeriod03 = counter + 1
            break
        counter = counter + 1
        frameTimeP = frameTime

    el_definePeriods = timemod.time() - st_definePeriods
    print('defining periods                 ' + str(round(el_definePeriods, 3)) + 's to complete')

    # pay attention while using these indices for defining periods
    # period n-1    : time[iPeriod01:iPeriod02 + 1]
    # period n      : time[iPeriod02 - 1:iPeriod03]
    return(iPeriod01, iPeriod02, iPeriod03)




















def interp1D1T(tw_start, tw_end, surfacePlane, TIME, omega):
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    #
    # ARGUMENTS
    #
    # xpoints       : resolution in the along channel direction
    # tw_start      : start index of the points along the Thalweg
    # tw_end        : end index of the points along the Thalweg
    # surfacePlane  : 2d matrix, with the first dimension referring to time and
    #                 the second dimension referring to the horizontal distribution of the points
    # TIME          : 1d vector containing times spreading over at least one period
    # omega         : leading order frequency of the signal
    #
    # RETURNS
    #
    # time_interp   : time vector covering one single period with an integer number N time intervals
    # wl_interp     : matrix (x, time) containing the interpolated values for the water-level along estuary and over one period
    #
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################


    # interpolate the time with regularly spaced intervals AND an integer number of intervals within one period
    N = 90    # define number of points per period
    period = 2. * np.arccos(-1.) / omega
    periodStart = round(TIME[0] / period) * period  # determine the starting time of a full period
    time_interp = np.asarray([periodStart+float(count) / float(N) * period for count in
                                  range(0, int((TIME[-1]-TIME[0])/period) * N)])

    wl_interp=np.zeros([tw_end-tw_start, N])


    for iPoint in range(tw_start, tw_end):
        i = iPoint - tw_start
        timeseries = surfacePlane[:, iPoint]

        # interpolate the results on the new time vector
        interpfunction = interp1d(TIME, timeseries, kind='cubic')
        wl_interp[i, :] = interpfunction(time_interp)

    return(time_interp, wl_interp)


def CalculateM2M4_FFT(surfacePlane, time_interp, omega):
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################


    # surfacePlane  : 2d matrix, with the first dimension referring to the horizontal distribution of the points and
    #                 the second dimension referring to time
    # TIME          : 1d vector containing times spreading over at least one period
    # omega         : leading order frequency of the signal

    N=len(time_interp)
    xpoints=np.shape(surfacePlane)[0]


    M2_thalweg = np.zeros([xpoints, 3])
    M4_thalweg = np.zeros([xpoints, 3])
    M6_thalweg = np.zeros([xpoints, 3])
    M8_thalweg = np.zeros([xpoints, 3])

    # python starts from ip_start-1 to ip_end-1
    for iPoint in range(0, xpoints):

        wl_interp=surfacePlane[iPoint, :]

        freq = 2 * np.pi * fftfreq(N, 1. / float(N) * 2. * np.arccos(-1.) / omega)
        sp = fft(wl_interp)

        amp = 2.0 / N * np.abs(sp)
        phi = np.angle(sp) * 180 / np.pi  # phase in degree

        # get the index of M2, M4, M6 and M8
        id_m2 = (np.abs(freq - omega)).argmin()
        id_m4 = (np.abs(freq - 2 * omega)).argmin()
        id_m6 = (np.abs(freq - 3 * omega)).argmin()
        id_m8 = (np.abs(freq - 4 * omega)).argmin()

        # get the amp, freq and phase of each component
        M2_thalweg[iPoint, :] = [freq[id_m2], amp[id_m2], phi[id_m2]]
        M4_thalweg[iPoint, :] = [freq[id_m4], amp[id_m4], phi[id_m4]]
        M6_thalweg[iPoint, :] = [freq[id_m6], amp[id_m6], phi[id_m6]]
        M8_thalweg[iPoint, :] = [freq[id_m8], amp[id_m8], phi[id_m8]]

    #todo: write the output in an iflow way, i.e. one single vector containing the complex M2, M4, M6, M8 signals
    return (M2_thalweg, M4_thalweg, M6_thalweg, M8_thalweg)


