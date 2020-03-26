import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# read the estuary data

estuaryfileName = 'C:\\Users\\saaad264\\Research\\16_072_LIFE_sparc\\C10_Idealised_modelling\\estuary_geometry\\Schelde\\geometry_bathymetry.txt'
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

popt_w, pcov_w = curve_fit(funcWidth, dist_data[dist_data>0], width_data[dist_data>0])

# compute the RMSE and R2
ydata = width_data[dist_data>0]
xdata = dist_data[dist_data>0]
RMSE = np.square(np.subtract(ydata,funcWidth(xdata, *popt_w))).mean()
residuals = ydata - funcWidth(xdata, *popt_w)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)
r_squared = 1 - (ss_res / ss_tot)

# plot
plt.figure(figsize=(8,4))
plt.plot(dist_data[dist_data>0], width_data[dist_data>0], 'ro',markersize=3)
plt.plot(dist_data[dist_data>0], funcWidth(dist_data[dist_data>0], *popt_w), 'b-')
plt.text(80, 4000, "$R^{2}$="+str(r_squared))
plt.title('Representative width along the Scheldt Estuary')
plt.xlim([0,160])
plt.ylim([0,8000])
plt.xlabel('Distance from the estuary mouth (km)')
plt.ylabel('Estuary width (m)')
plt.grid(which='both',axis='both',color='lightgrey')
# plt.show()
plt.savefig('Representative width along the Scheldt Estuary'+'.png', dpi=300)

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

popt_b, pcov_b = curve_fit(funcBottom, dist_data[dist_data>0], bottom_data[dist_data>0])

# compute the RMSE and R2
ydata = bottom_data[dist_data>0]
xdata = dist_data[dist_data>0]
RMSE = np.square(np.subtract(ydata,funcBottom(xdata, *popt_b))).mean()
residuals = ydata - funcBottom(xdata, *popt_b)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)
r_squared = 1 - (ss_res / ss_tot)

plt.figure(figsize=(8,4))
plt.plot(dist_data[dist_data>0], bottom_data[dist_data>0], 'ro',markersize=3)
plt.plot(dist_data[dist_data>0], funcBottom(dist_data[dist_data>0], *popt_b), 'b-')
plt.text(10, -3, "$R^{2}$="+str(r_squared))
plt.title('Representative bottom elevation along the Scheldt Estuary')
plt.xlim([0,160])
plt.ylim([-18,0])
plt.xlabel('Distance from the estuary mouth (km)')
plt.ylabel('Bottom elevation (m MSL)')
plt.grid(which='both',axis='both',color='lightgrey')
# plt.show()
plt.savefig('Representative bottom elevation along the Scheldt Estuary'+'.png', dpi=300)

