'''
 modelling a RC-based XOR gate, with 2 input signals of different phases and amplitudes 
 and a binary output
'''  

import numpy as np
from scipy import linalg as LA


# setting the integration time and time-step:
dt = 0.001
tmax = 15 #input("How long (in seconds) should the system run? ")
tmax = float(tmax)
tlist = np.arange(0, tmax, dt)
nt = tlist.size

# setting the node variable (firing rates): 
#X = np.full((N, nt), 0.0)
## loading the Reservoir's Adjacency Matrix from Memory:
W = np.load('AdjacencyMatrix_N=10_max(SV)=0.933745180396509.npy')
print(max("spectral radius", LA.eigvals(W)))

N = W.shape[0]
print(f"Reservoir's Size = {N}")
X = np.zeros((N,nt))
X_intr = np.zeros((N,nt))

# Teacher Signal
X_trg = np.loadtxt('XOR_sequence..txt')[:,0:1]
trn_length = X_trg.shape[0]



# initiating the vector field:
F = np.zeros((N,nt)) 
F1 = np.zeros((N,nt))
F2 = np.zeros((N,nt))


x = np.zeros(N)
x_intro = np.zeros(N)
f_1 = np.zeros(N)
f_2 = np.zeros(N)

# running the reservoir in training mode
for i in range(1, trn_length): # Heun's Method
	f_1 = (W @ x) + U[:,i-1]#.reshape((10,1))
	x_intr = (x + dt * f_1)
	f_2 = (W @ x_intr) + U[:,i]#.reshape((10,1))
	x = x + dt * (f_1 + f_2) / 2
	X[:,i] = x

# read-outs functions:
def H(x):
	if x < 0:
		return 0
	else:
		return 1

def sgm(x):	
	return 1 / (1 + np.exp(-x))






M = 1 # dimension of Y:
Y = np.zeros((M,nt))
UX = np.concatenate((D,X))
Y = sgm(W_out @ UX)

# teacher output signal:
Y_trg = np.loadtxt('XOR_sequence..txt')[:,2] #0 for any U[:] == 0

# training the W_out, by comparing the out-put with teacher-set:
W_out = np.random.random((M,N))
W_out = Y_trg @ LA.inv(X)

# computing the MSE:
E = (Y_trg - Y)**2


# Input Signal(s):
A = np.zeros(N)
#A[3] = 0.03
A[9] = 0.15

FR = np.zeros(N)
FR[9]= 40.0

PHI = np.zeros(N)
#PHI[3] = np.pi / 2
PHI[9] = 5.0

t1 = 1.2 # Input's starttime
t2 = 5.0 # Input's endtime
ti_1 = int(t1 / dt)
ti_2 = int(t2 / dt)
U = np.zeros((N,nt))

for i in range(N):
	U [i,ti_1:ti_2] = A[i] * np.sin(FR[i] * tlist[ti_1:ti_2] + PHI[i])

# running the reservoir after training the read-out weights
for i in range(1,nt): # Heun's Method
	f_1 = (W @ x) + U[:,i-1]#.reshape((10,1))
	x_intr = (x + dt * f_1)
	f_2 = (W @ x_intr) + U[:,i]#.reshape((10,1))
	x = x + dt * (f_1 + f_2) / 2
	X[:,i] = x

# storing the results:

# data = {'time': tlist, 'response': X[i,:]}
# df = pd.DataFrame(data) 
# df.to_csv('results.csv')

