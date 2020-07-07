
import numpy as np
from scipy import linalg as LA


# setting the integration time and time-step:
dt = 0.001
tmax = 30 #input("How long (in seconds) should the system run? ")
tmax = float(tmax)
tlist = np.arange(0, tmax, dt)
nt = tlist.size

# setting the node variable (firing rates): 
#X = np.full((N, nt), 0.0)
W = 10 * np.load('AdjacencyMatrix_N=10_max(SV)=0.933745180396509.npy')
#W = np.load('AdjacencyMatrix_N=200_max(SV)=0.798523507401671.npy')
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

drived_node = int(input("which node to drive? "))
A = np.zeros(N)
#A[3] = 0.03
A[drived_node] = 0.15

FR = np.zeros(N)
FR[drived_node] = 10.0 #40.0

PHI = np.zeros(N)
#PHI[3] = np.pi / 2
PHI[drived_node] = 1.571 #np.pi / 2

t1 = 1.2 # Input's start time
t2 = 15.0 # Input's end time
ti_1 = int(t1 / dt)
ti_2 = int(t2 / dt)
U = np.zeros((N,nt))

for i in range(N):
	U [i,ti_1:ti_2] = A[i] * np.sin(FR[i] * tlist[ti_1:ti_2] + PHI[i])


# initiating the vector field:
F = np.zeros((N,nt)) 
F1 = np.zeros((N,nt))
F2 = np.zeros((N,nt))
## loading the Adjacency Matrix from Memory:
# Temp = pd.read_csv('AdjacencyMatric_N=8_maxSV=0.9527945058321626.csv')
# W = Temp[:,1:]

EVs, V = LA.eig(W)
print(max(EVs))
print(min(EVs))
V_inv = LA.inv(V)
U_new = np.real(V @ U)
# W1 = V_inv @ W @ V
# print(W1)
W_new = np.diag(EVs)
print(W_new)
# print(np.allclose(W1,W2))

negative_inverse_tau = set(np.diagonal(W))
print(negative_inverse_tau)
print("Integration Started...")
for i in range(1,nt): # Heun's Method
	F[:,i-1] = W_new @ X[:,i-1] + U_new[:,i-1] 
	X_intr[:,i] = X[:,i-1] + dt * F[:,i-1]  
	F[:,i] = W_new @ X_intr[:,i] + U_new[:,i]
	X[:,i] = X[:,i-1] + dt * (F[:,i-1] + F[:,i]) / 2

# recasting the response in the eigenbasis of Reservoir
V_inv = V_inv.real
X_new = np.zeros((N,nt))
for i in range(nt):
	X_new[:,i] = V_inv @ X[:,i]

# storing the results:


# data = {'time': tlist, 'response': X[i,:]}
# df = pd.DataFrame(data) 
# df.to_csv('results.csv')

# visualizations:
from matplotlib import pyplot as plt 



plt.xlabel("time (sec)")
plt.ylabel("Nodes Responses")

'''
ax1 = plt.subplot(211)
plt.title("Input Signal")
for i in range(N):
	plt.plot(tlist, U[i,:], label = f'{i}')

ax2 = plt.subplot(212, sharey = ax1)
'''
#for i in range(N):
#	print("%.3f" % EVs[i])

plt.title(f"Linear Reservoir of {N} nodes by driving node#{drived_node}\n with the Input signal of Amplitude {float(A[np.where(A!=0)])}, frequency {float(FR[np.where(FR!=0)])} Hz & phase $\pi$/2")
# plt.subplot(2,1,1)
for i in range(N):
	plt.plot(tlist, X[i,:], label = f'$\lambda_{i}$: {"%.4f" % EVs[i]}')
# plt.subplot(2,1,2)
# for i in range(N):
# 	plt.plot(tlist, X_new[i,:], label = f'$\lambda_{i}$: {EVs[i]}')
	#plt.plot(tlist, U[i,:], '--')
plt.legend(fontsize='x-small',loc = 'upper right')

plt.savefig(f"LR_N={N}.svg",format='svg')
plt.show()
