import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# read the estuary data

estuaryfileName = 'C:\\Users\\saaad264\\Research\\16_072_LIFE_sparc\\estuary_geometry\\Ems\\geometry_bathymetry.txt'
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
dist_data  = dist [id_width] # the data is in (m)
width_data = width[id_width]

# define curve functions

# B = 762 - 4.18e-2*x + 3.83e-6*x^2 – 1.77e-10*x^3 + 3.27e-15*x^4 – 2.05e-20*x^5

# def funcWidth(x):
#     return 762.0 - 4.18e-2*x + 3.83e-6*x**2.0 - 1.77e-10*x**3.0 + 3.27e-15*x**4.0 - 2.05e-20*x**5.0

def funcWidth(x,c1,c2,c3,c4,c5,c6):
    return np.polyval([c1,c2,c3,c4,c5,c6],x/1000)

popt_w, pcov_w = curve_fit(funcWidth, dist_data[dist_data>0], width_data[dist_data>0])

# compute the RMSE
width_fitting_RMSE = np.square(np.subtract(width_data[dist_data>0],funcWidth(dist_data[dist_data>0], *popt_w))).mean()
# width_fitting_RMSE = np.square(np.subtract(width_data[dist_data>0],funcWidth(dist_data[dist_data>0]))).mean()

# plot
plt.figure(figsize=(8,4))
plt.plot(dist_data[dist_data>0]/1000, width_data[dist_data>0], 'ro',markersize=3)
plt.plot(np.arange(0, 64.01, 0.01), funcWidth(np.arange(0, 64.01, 0.01)*1000, *popt_w), 'b-')
plt.text(30, 700, "RMSE="+str(width_fitting_RMSE))
plt.title('Representative width along the Ems Estuary')
plt.xlim([0,64])
plt.ylim([0,1000])
plt.xlabel('Distance from the estuary mouth (km)')
plt.ylabel('Estuary width (m)')
plt.grid(which='both',axis='both',color='lightgrey')
plt.savefig('Representative width along the Ems Estuary'+'.png', dpi=300)
plt.show()

# for bottom ########################
id_bottom = ~np.isnan(bottom)
dist_data  = dist[id_bottom]
bottom_data = 0 - bottom[id_bottom]

# define curve functions

# H = -0.5*1.2*(1+tanh((x-13000)/5000)) - 0.5*5.1e-5*(1+tanh((x-13000)/5000))*x + 10

def funcBottom(x):
    return -0.5*1.2*(1+np.tanh((x-13000)/5000)) - 0.5*5.1e-5*(1+np.tanh((x-13000)/5000))*x + 10

# def funcBottom(x,c1,c2,c3):
#     return c1*(1+np.tanh((x-13000)/5000)) - c2*(1+np.tanh((x-13000)/5000))*x + c3

# def funcBottom(x,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10):
#     return np.polyval([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10],x)

popt_b, pcov_b = curve_fit(funcBottom, dist_data[dist_data>0], bottom_data[dist_data>0])

# compute the RMSE
# bottom_fitting_RMSE = np.square(np.subtract(bottom_data[dist_data>0],funcBottom(dist_data[dist_data>0], *popt_b))).mean()

bottom_fitting_RMSE = np.square(np.subtract(bottom_data[dist_data>0],funcBottom(dist_data[dist_data>0]))).mean()

plt.figure(figsize=(8,4))
plt.plot(dist_data[dist_data>0]/1000, bottom_data[dist_data>0], 'ro',markersize=3)
# plt.plot(np.arange(0, 64.01, 0.01), -funcBottom(np.arange(0, 64.01, 0.01)*1000, *popt_b), 'b-')
plt.plot(np.arange(0, 64.01, 0.01), -funcBottom(np.arange(0, 64.01, 0.01)*1000), 'b-')
plt.text(5, -4, "RMSE="+str(bottom_fitting_RMSE))
plt.title('Representative bottom elevation along the Ems Estuary')
plt.xlim([0,64])
plt.ylim([-12,-2])
plt.xlabel('Distance from the estuary mouth (km)')
plt.ylabel('Bottom elevation (m MSL)')
plt.grid(which='both',axis='both',color='lightgrey')
# plt.savefig('Representative bottom elevation along the Ems Estuary'+'.png', dpi=300)
plt.show()

