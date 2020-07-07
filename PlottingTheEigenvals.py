
import numpy as np
from scipy import linalg as LA
from matplotlib import pyplot as plt
import os


my_path = os.path.abspath('Generated_Reservoirs')
print(my_path)

# W = 1 * np.load('AdjacencyMatrix_N=10_max(SV)=0.933745180396509.npy')
# W = 1 * np.load('AdjacencyMatrix_N=10_max(SV)=0.8664873639666085.npy')
# W = 1 * np.load('AdjacencyMatrix_N=10_max(SV)=0.8728963013881634.npy')
# W = 1 * np.load('AdjacencyMatrix_N=10_max(SV)=0.8892999419073212.npy')
# W = 1 * np.load('AdjacencyMatrix_N=10_max(SV)=0.9195179062090398.npy')
# W = 1 * np.load('AdjacencyMatrix_N=10_max(SV)=0.9247961869114325.npy')
# W = 1 * np.load('AdjacencyMatrix_N=10_max(SV)=0.9677920512563868.npy')
W = 1 * np.load(f'{my_path}\AdjacencyMatrix_N=50_max(SV)=0.9387091213578811.npy')


EVs, V = LA.eig(W)

N = W.shape[0]

plt.figure(figsize=[14,6.5])
plt.xlabel("real part")
plt.ylabel("imaginary part")
plt.title("distribution of EVs")
plt.xlim(-1,0)
plt.ylim(-1,1)

# for i in range(N):
plt.subplot(2,1,1)
plt.title('Matrix Elements')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.scatter(W, np.zeros(N**2), color='b')

plt.subplot(2,1,2)
plt.title('Eigenvalues')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.scatter(np.real(EVs),np.imag(EVs), color='k')

#plt.legend(fontsize='x-small',loc = 'upper right')
plt.show()

