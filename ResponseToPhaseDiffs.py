
'''
estimating the node responses in a 2pi period of phase shift in the input signal
'''

import numpy as np
# from scipy import linalg as LA
import os
from matplotlib import pyplot as plt 
# import pandas as pd

N = 10
os.chdir('the_Phase_Effect_Lab')
X_maxes = np.zeros(N).reshape(N,1)
print(X_maxes.shape)

for file in os.listdir():
    _ , f_ext = os.path.splitext(file)
    if f_ext == ".npy":
        X = np.load(file)
        temp = np.amax(X, axis=1).reshape((N,1))
        # print(temp)
        # print(temp.shape)
        X_maxes = np.hstack((X_maxes, temp))
        f_name, _ = os.path.splitext(file)
name_list = f_name.split('_') #to extract info from the file name
# N = name_list[2].split('=')[1]
# N = int(N)
print(name_list)       
print(temp.shape)

# print(X_maxes.size)
# for j in range(32): 
#     print(X_maxes[:,j], j)
X_maxes = np.delete(X_maxes, 0, axis=1)
print(X_maxes.shape)
# print(X_maxes)
# np.save(f'ResponseAmps#{name_list[1]}_N_{N}_DrivedNodes{name_list[-2]}&{name_list[-1]}.npy', X_maxes)
np.savetxt(f'ResponseAmps#{name_list[1]}_N_{N}_DrivedNodes{name_list[-2]}&{name_list[-1]}.txt', X_maxes)
# np.load('ResponseAmps.npy')


mins = np.amin(X_maxes, axis=1)
maxes = np.amax(X_maxes, axis=1)
# Responsivity = (maxes - mins) / np.amax(A)
Responsiveness = maxes - mins
for i in range (N):
    print(i, "%.4f" % Responsiveness[i])
  
## Visualisations
plt.figure(figsize=[14,6.5])
plt.title(f'Responses for the Reservoir #{name_list[1]}, Drived Nodes {name_list[-2]} & {name_list[-1]}')
plt.xlabel('Node Labels')
plt.ylabel('Range of Response Amplitudes')
plt.xticks(range(N), fontsize='small', rotation=45)
# plt.ylim(0,0.005)
plt.grid(axis='x')
plt.scatter(np.arange(N), maxes, label = 'Max Response Amplitude')
plt.scatter(np.arange(N), mins, label = 'Min Response Amplitude')
plt.legend(loc='upper right')
plt.savefig(f'#{name_list[1]}_N_{N}_DrivedNodes{name_list[-2]}&{name_list[-1]}.png')
plt.show()


