import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# read the estuary data

estuaryfileName = 'C:\\Users\\saaad264\\Research\\16_072_LIFE_sparc\\estuary_geometry\\Schelde\\geometry_bathymetry.txt'
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
def funcWidth(x,c11,c12,c21,c22,c23):
    return 1000.0*np.exp(np.divide(np.polyval([c11,c12],x),np.polyval([c21,c22,c23],x)))

popt, pcov = curve_fit(funcWidth, dist_data[dist_data>0], width_data[dist_data>0])

# compute the RMSE
width_fitting_RMSE = np.square(np.subtract(width_data[dist_data>0],funcWidth(dist_data[dist_data>0], *popt))).mean()

# plot
plt.plot(dist_data[dist_data>0], width_data[dist_data>0], 'o')
plt.plot(dist_data[dist_data>0], funcWidth(dist_data[dist_data>0], *popt), 'r-')
plt.text(80, 4000, "RMSE="+str(width_fitting_RMSE))
plt.show()

# for bottom ########################
id_bottom = ~np.isnan(bottom)
dist_data  = dist[id_bottom]
bottom_data = bottom[id_bottom]

# define curve functions

def funcBottom(x,c1,c2,c3,c4,c5):
    XL = 130.0
    y1 = np.polyval([c1,c2,c3,c4,c5],x[x<XL])
    y2 = np.polyval([c1,c2,c3,c4,c5],XL) + np.polyder(np.poly1d([c1,c2,c3,c4,c5]))(XL)*(x[x>=XL]-XL)
    return np.concatenate((y2, y1), axis=None) # in case x has descending order

XL = 130.0

popt, pcov = curve_fit(funcBottom, dist_data[dist_data>0], bottom_data[dist_data>0])

# compute the RMSE
bottom_fitting_RMSE = np.square(np.subtract(bottom_data[dist_data>0],funcBottom(dist_data[dist_data>0], *popt))).mean()

plt.plot(dist_data[dist_data>0], bottom_data[dist_data>0], 'o')
plt.plot(dist_data[dist_data>0], funcBottom(dist_data[dist_data>0], *popt), 'r-')
plt.text(10, -3, "RMSE="+str(bottom_fitting_RMSE))
plt.show()


