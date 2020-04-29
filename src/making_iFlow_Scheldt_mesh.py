import numpy as np
import math
import matplotlib.pyplot as plt

# The shape of the idealized Scheldt estuary
# (Dijkstra et al. 2019_OD69)

x = np.linspace(0, 160000, 1601)
C1 = np.array([-0.02742e-3, 1.8973])
C2 = np.array([4.9788e-11, -9.213e-6, 1.0])

B0 = 1000*np.exp(np.polyval(C1, x)/np.polyval(C2, x))
plt.plot(x, B0, x, -B0)
plt.show()

XL = 1.29069757e+05
C = np.array([3.72661847e-19, -1.15133973e-13,
              1.13427367e-08, -4.34610842e-04, 1.50000000e+01])
H0 = np.zeros([len(x)])

H0_1 = np.asarray([np.polyval(C, i) for i in x if i < XL])
p2 = np.polyder(np.poly1d(C))
H0_2 = np.asarray([np.polyval(C, XL)+p2(XL)*(i-XL) for i in x if i >= XL])
H0 = np.concatenate((-H0_1, -H0_2), axis=None)
plt.plot(x, H0)
plt.show()

Scheldt_shape = np.concatenate((np.reshape(x, (x.shape[0], 1)),np.reshape(B0, (B0.shape[0], 1)),np.reshape(-B0, (B0.shape[0], 1)),np.reshape(H0, (H0.shape[0], 1))), axis=1)
np.savetxt('idealized_Scheldt_shape_100.txt',Scheldt_shape, delimiter=',',header='[X,B0,-B0,H0]')

# Calculate mesh size field
n = 10
S0 = np.asarray([i/2 for i in B0])
CS = np.polyfit(x, S0, n)
SF = np.poly1d(np.polyfit(x, S0, n))

expression = ""
for i in range(n+1):
    expression+=str(CS[i])+'*x^'+str(n-i)+'+'

plt.plot(x,S0,x,SF(x))
plt.show()

# size(x) = (-2.9911384871301455e-47*x^10+2.608079234878421e-41*x^9-9.539488276486646e-36*x^8+1.8753189389846865e-30*x^7-2.0932729126329697e-25*x^6+1.273751227939705e-20*x^5-3.8423573237245636e-16*x^4+1.0743635492411506e-11*x^3-5.543335697358514e-07*x^2-0.032690493317448735*x^1+3333.4419446766206)

# check distance between two points

# Tidal forcing
PI = np.arccos(-1.0)

A_M2 = 1.70378540307085
A_M4 = 0.124747864259366
A_S2 = 0.458591288878057

PHI_M2 = (30.8259062310508-30.8259062310508)*PI/180.0
PHI_M4 = (60.9361966955613-30.8259062310508)*PI/180.0
PHI_S2 = (88.2287441596619-30.8259062310508)*PI/180.0

FREQ_M2 = 2.0*PI/(3600.0*12.4206)
FREQ_M4 = 2.0*FREQ_M2
FREQ_S2 = 2.0*PI/(3600.0*12.00)

TIME = np.arange(0,2592000,600)

SL3 = A_M2*np.cos(FREQ_M2*TIME+PHI_M2)+ A_M4*np.cos(FREQ_M4*TIME+PHI_M4)+ A_S2*np.cos(FREQ_S2*TIME+PHI_S2)