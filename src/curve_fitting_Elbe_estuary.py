import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# read the estuary data

estuaryfileName = 'C:\\Users\\saaad264\\Research\\16_072_LIFE_sparc\\estuary_geometry\\Elbe\\geometry_bathymetry.txt'
estuaryfile = open(estuaryfileName, 'r')
lines = estuaryfile.readlines()

tmpData = []

for lineStr in lines:
    lineStr = lineStr.rstrip().split('\t')
    lineNum = [float(lineStr[i]) for i in range(3)]
    tmpData.append(lineNum)

# data format: [Distance width bottom]
estuaryData = np.array(tmpData)
# change the descending order to ascending order
dist =   estuaryData[:,0]
width =  estuaryData[:,1]
bottom = estuaryData[:,2]

# remove nan values then perform curve fitting

# for width ########################
id_width = ~np.isnan(width)
dist_data  = dist [id_width]
width_data = width[id_width]

# define curve functions

# def funcWidth(x,c0,c1,c2,c3,c4,c5,c6,c7,c8):
#     return np.polyval([c0,c1,c2,c3,c4,c5,c6,c7,c8],x)

# def funcWidth(x,c11,c12,c21,c22,c23):
#     return 1000.0*np.exp(np.divide(np.polyval([c11,c12],x),np.polyval([c21,c22,c23],x)))

# def funcWidth(x,c0,Lc):
#     return c0*np.exp(-x/Lc)

def funcWidth(x,c0,c1,xc,xl):
    return c0+c1*np.tanh((x-xc)/xl)

popt, pcov = curve_fit(funcWidth, dist_data[dist_data>0], width_data[dist_data>0])

# compute the RMSE
width_fitting_RMSE = np.square(np.subtract(width_data[dist_data>0],funcWidth(dist_data[dist_data>0], *popt))).mean()

plt.plot(dist_data[dist_data>0], width_data[dist_data>0], 'o')
plt.plot(dist_data[dist_data>0], funcWidth(dist_data[dist_data>0], *popt), 'r-')
plt.text(50, 3200, "RMSE="+str(width_fitting_RMSE))
plt.show()

# for bottom ########################
id_bottom = ~np.isnan(bottom)
# id_bottom = np.logical_and(~np.isnan(bottom),dist>0)
dist_data  = dist[id_bottom]
bottom_data = bottom[id_bottom]

# define curve functions

def funcBottom(x,c00,c10,c20,c30,c40,c01,c11,c21,c31,c41,c51,c61,c02,c12,c22,c32,c42,c03,c13,c23):
    XL0 = 20.0
    XL1 = 66.0
    XL2 = 82.0
    y0 = np.polyval([c00,c10,c20,c30,c40],x[x<XL0])
    y1 = np.polyval([c01,c11,c21,c31,c41,c51,c61],x[np.logical_and(x>=XL0,x<XL1)])
    y2 = np.polyval([c02,c12,c22,c32,c42],x[np.logical_and(x>=XL1,x<XL2)])
    y3 = np.polyval([c03,c13,c23],x[x>=XL2])
    return np.concatenate((y3, y2, y1, y0), axis=None) # in case x has descending order

popt, pcov = curve_fit(funcBottom, dist_data[dist_data>0], bottom_data[dist_data>0])

# compute the RMSE
bottom_fitting_RMSE = np.square(np.subtract(bottom_data[dist_data>0],funcBottom(dist_data[dist_data>0], *popt))).mean()

plt.plot(dist_data[dist_data>0], bottom_data[dist_data>0], 'o')
plt.plot(dist_data[dist_data>0], funcBottom(dist_data[dist_data>0], *popt), 'r-')
plt.text(5, -16, "RMSE="+str(bottom_fitting_RMSE))
plt.show()


