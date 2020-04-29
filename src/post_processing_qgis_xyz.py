import numpy as np
import matplotlib.pyplot as plt
import os

# read .xyz file
working_dir = 'D:\\QILONG_BI\\Research\\13_131_C Scenarios\\Uitwisseling bathymetrie C alternatieven\\v6\\'
file_name = 'C2_v6_RD_New.xyz'

file_path = working_dir + file_name
out_path = working_dir + file_name.replace('.xyz','_clipped.xyz')

no_data = -3.4028231e+38

outfile = open(out_path,'a')

with open(file_path,'r') as f:
    for line in f:
        line = f.readline().strip().split(' ')
        if float(line[-1]) != no_data:
            # print(line)
            _ = outfile.write(' '.join(line)+'\n')

outfile.close()