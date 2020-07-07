
import numpy as np
from scipy import linalg as LA

'''
N = 100 # nodes number

# connectivity matrix:
T = np.random.uniform(-1,1,(N,N))
# for i in range(N):
	# T[i,i] = 2.5

# making the network sparse: 	
for i in range (int(0.80 * N**2)):
	m = np.random.randint(0,N)
	n = np.random.randint(0,N)
	T[m,n] = 0.0

# ensuring the connectedness of the whole network:

# singular value of T < 1


# ensuring statibility of the FPs:	
np.fill_diagonal(T, -7.0)
#T = np.tril(T,k=0)



#T = -1 * T
# ensuring the stability of the FPs:
l = LA.eigvals(T)
T = 0.97 * (T / max(abs(l)))
print(T)
l, v = LA.eig(T)

#print(l, v)

# ensuring ESP?! How?

# ESP condition: max(LA.svdvals(T)) < 1
'''

# setting the integration time and time-step:
dt = 0.001
tmax = 15 #input("How long (in seconds) should the system run? ")
tmax = float(tmax)
tlist = np.arange(0, tmax, dt)
nt = tlist.size

# loading the graph
T = np.load('AdjacencyMatrix_N=100_max(SV)=0.8196344987253827.npy')
N = T.shape[0]
print(f'N = {N}')

# setting the node variable (firing rates): 
#X = np.full((N, nt), 0.0)
X = np.zeros((N,nt))
#X[:,0] = 0.1
X_intr = np.zeros((N,nt))
X1 = np.zeros((N,nt))
#X1[:,0] = 0.1
X1_intr = np.zeros((N,nt))
X2 = np.zeros((N,nt))
#X2[:,0] = 0.1
X2_intr = np.zeros((N,nt))

# initial perturbation:
#X[:,0] = np.random.random(N)
# X[0,0] = 0.1


# external drive:	
D1 = np.zeros((N,nt))
D2 = np.zeros((N,nt))
L = 7000 # Input's length
D1[3,0:L] = 0.03 #* np.sin(5*tlist[0:L]) 
D1[4,0:L] = 0.15 * np.sin(40 * tlist[0:L])
D2[4,0:L] = 0.02 * np.sin(25 * tlist[0:L] + 1)
D = D1 + D2

F = np.zeros((N,nt)) #T @ X + D
F1 = np.zeros((N,nt))
F2 = np.zeros((N,nt))

print('integrating the response to the 1st input')
for i in range(1,nt):
	#X_dot = T @ X + D # computationally very expensive! but also wrong(?)
	F1[:,i-1] = T @ X1[:,i-1] + D1[:,i-1] 
	X1_intr[:,i] = X1[:,i-1] + dt * F1[:,i-1]  
	F1[:,i] = T @ X1_intr[:,i] + D1[:,i]
	X1[:,i] = X1[:,i-1] + dt * (F1[:,i-1] + F1[:,i]) / 2

print('integrating the response to the 2nd input')
for i in range(1,nt):
	#X_dot = T @ X + D # computationally very expensive! but also wrong(?)
	F2[:,i-1] = T @ X2[:,i-1] + D2[:,i-1] 
	X2_intr[:,i] = X2[:,i-1] + dt * F2[:,i-1]  
	F2[:,i] = T @ X2_intr[:,i] + D2[:,i]
	X2[:,i] = X2[:,i-1] + dt * (F2[:,i-1] + F2[:,i]) / 2

print('integrating the response to the composite input')
for i in range(1,nt): # Heun's Method

	F[:,i-1] = T @ X[:,i-1] + D[:,i-1] 
	X_intr[:,i] = X[:,i-1] + dt * F[:,i-1]  
	F[:,i] = T @ X_intr[:,i] + D[:,i]
	X[:,i] = X[:,i-1] + dt * (F[:,i-1] + F[:,i]) / 2
	
X_sum = X1 + X2

print('is the response linear?', X_sum == X)

	
# visualizations:
from matplotlib import pyplot as plt 
#plt.subplot(1,1)
plt.title("Linear Reservoir for N={N}")
plt.xlabel("time (sec)")
plt.ylabel("Nodes Responses")

# ax1 = plt.subplot(411)
# for i in range(N):
	# plt.plot(tlist, X1[i,:], label = f'{i}')

# ax2 = plt.subplot(412, sharex = ax1, sharey = ax1)
# for i in range(N):
	# plt.plot(tlist, X2[i,:], label = f'{i}')


ax1 = plt.subplot(211)
for i in range(N):
	plt.plot(tlist, X_sum[i,:], label = f'{i}')

plt.title(f"sum of the Responses to separate inputs, N={N}")
#plt.xlabel("time (sec)")
plt.ylabel("Nodes Responses")

ax2 = plt.subplot(212, sharex = ax1, sharey = ax1)
for i in range(N):
	plt.plot(tlist, X[i,:], label = f'{i}')
#plt.plot(tlist, D[4,:],'k')
	
#plt.legend(loc = 'lower right')
#plt.subplot(2,1)
plt.title(f"Response to sum of the inputs, N={N}")
plt.xlabel("time (sec)")
plt.ylabel("Nodes Responses")

plt.show()