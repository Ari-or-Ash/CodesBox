
import numpy as np
from scipy import linalg as LA



# setting the integration time and time-step:
dt = 0.001
tmax = 40 #input("How long (in seconds) should the system run? ")
tmax = float(tmax)
tlist = np.arange(0, tmax, dt)
nt = tlist.size

# setting the node variable (firing rates): 
#X = np.full((N, nt), 0.0)
W = 0.1 * np.load('AdjacencyMatrix_N=10_max(SV)=0.933745180396509.npy')
# W = 100 * np.load('AdjacencyMatrix_N=200_max(SV)=0.798523507401671.npy')
N = W.shape[0]
print(f'N = {N}')

X = np.zeros((N,nt))
X_intr = np.zeros((N,nt))


# initial perturbation:
#X[:,0] = np.random.random(N)
#X[0,0] = 0.1
#print(l)
#print(LA.svdvals(W))


# # external drive:
# I = np.zeros(N,dtype=float)
# # drivenNode = input("choose a node to be driven by an input signal(by labels from 0 to (N-1)): ") 
# drivenNode = 0 #int(drivenNode)
# # Ii = 0.1
# ti = 0
# Ii = 1 #np.sin(10*2*np.pi*ti)
# I[drivenNode] = Ii
# print (I) # a sanity check

# # for i in range(1,N):
    # # globals()['X_%s' % i] =


# external drive:	
# D1 = np.zeros((N,nt))
# D2 = np.zeros((N,nt))
# L = int(0.2 * nt) # Input's length
# D1[3,1000:L] = 0.03 #* np.sin(5*tlist[0:L]) 
# D1[4,1000:L] = 0.15 * np.sin(40 * tlist[1000:L])
# D2[4,2500:L] = 0.02 * np.sin(25 * tlist[2500:L] + 1)
# D = D1 + D2

# 1st signal
drived_node1 = int(input("which node to drive? "))
A1 = np.zeros(N)
#A1[3] = 0.03
A1[drived_node1] = 0.0015

FR1 = np.zeros(N)
FR1[drived_node1]= 1.0 #40.0

PHI1 = np.zeros(N)
#PHI1[3] = np.pi / 2
PHI1[drived_node1] = 1.571 #np.pi / 2

# 2nd signal
drived_node2 = int(input("which other node to drive? "))
A2 = np.zeros(N)
#A2[3] = 0.03
A2[drived_node2] = A1[drived_node1] # 0.15

FR2 = np.zeros(N)
FR2[drived_node2]= FR1[drived_node1] #3.0 #40.0

PHI2 = np.zeros(N)
#PHI1[3] = np.pi / 2
PHI2[drived_node2] = -1.571 #np.pi / 2


t1 = 1.2 # Input's start time
t2 = 20.0 # Input's end time
ti_1 = int(t1 / dt)
ti_2 = int(t2 / dt)
U = np.zeros((N,nt))

for i in range(N):
	U [i,ti_1:ti_2] = A1[i] * np.sin(FR1[i] * tlist[ti_1:ti_2] + PHI1[i]) + \
		              A2[i] * np.sin(FR2[i] * tlist[ti_1:ti_2] + PHI2[i])
 
# initiating the vector field:
F = np.zeros((N,nt)) 

## loading the Adjacency Matrix from Memory:
# Temp = pd.read_csv('AdjacencyMatric_N=8_maxSV=0.9527945058321626.csv')
# W = Temp[:,1:]

EVs, V = LA.eig(W)
print(max(EVs))
print(min(EVs))
# V_inv = LA.inv(V)
# U_new = V @ U

negative_inverse_tau = set(np.diagonal(W))
print(negative_inverse_tau)
def sgm(x):	
	return 1 / (1 + np.exp(-x))

print("Integration Started...")
for i in range(1,nt): # Heun's Method
	F[:,i-1] = sgm(W @ X[:,i-1] + U[:,i-1]) 
	X_intr[:,i] = X[:,i-1] + dt * F[:,i-1]  
	F[:,i] = sgm(W @ X_intr[:,i] + U[:,i])
	X[:,i] = X[:,i-1] + dt * (F[:,i-1] + F[:,i]) / 2
	
# storing the results:
# import os
# my_path = os.path.abspath('checking_the_Phase_Effect')
# import pandas as pd
# time_stamps = {'time': tlist} 
# responses = {f'#{i}_response': X[i,:].round(6) for i in range(N)}
# data = {**time_stamps, **responses}
# df = pd.DataFrame(data) 
# PhaseDif = PHI2[drived_node2]-PHI1[drived_node1]
# df.to_csv(f'{my_path}/PhaseDif={"%.3f" % PhaseDif}_drived_nodes_#{drived_node1}_#{drived_node2}.csv')

# 
# plt.savefig(f"{my_path}/node_{drived_node}.png")
# plt.savefig(f"{my_path}/node_{drived_node}.svg")

# visualizations:
from matplotlib import pyplot as plt 
#plt.subplot(1,1)

plt.xlabel("time (sec)")
plt.ylabel("Nodes Responses")

'''
ax1 = plt.subplot(211)
plt.title("Input Signal")
for i in range(N):
	plt.plot(tlist, U[i,:], label = f'{i}')

ax2 = plt.subplot(212, sharey = ax1)
'''

plt.title(f"Linear Reservoir of {N} nodes by driving nodes #{drived_node1} & {drived_node2}\n with the Input signals of Amplitudes {float(A1[np.where(A1!=0)])}, frequencies {float(FR1[np.where(FR1!=0)])} Hz & phases {float(PHI1[np.where(PHI1!=0)])}&{float(PHI2[np.where(PHI2!=0)])}")
for i in range(N):
	plt.plot(tlist, X[i,:], label = f'$\lambda_{i}$: {"%.3f" % np.real(EVs[i])} + {"%.3f" % np.imag(EVs[i])}i')
	#plt.plot(tlist, U[i,:], '--')
plt.legend(fontsize='x-small',loc = 'upper right')
#plt.subplot(2,1)
# plt.savefig(f"LR_N={N}.svg",format='svg')
plt.show()