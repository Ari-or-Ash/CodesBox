
import numpy as np
from scipy import linalg as LA


N = 10 # nodes number


# connectivity matrix:
#T = np.random.random((N,N))
T = np.random.uniform(-1,1,(N,N))


# making the network sparse: 
s = 0.98	# sparcity parameter, (1 - density)
for i in range (int(s * N**2)): # i could improve this loop/process
	m = np.random.randint(0,N)
	n = np.random.randint(0,N)
	T[m,n] = 0.0


# ensuring statibility of the FPs:

Diagonal = - (np.sqrt(1 - s)) * N / 2
np.fill_diagonal(T, -50)#Diagonal)



# ensuring the connectedness of the whole network:
## ... comming soon!

# ensuring the existence of cycles:
## ... coming soon

# ensuring ESP?! How?
### ESP condition: max(LA.svdvals(T)) < 1

l = LA.eigvals(T)
T = 0.93 * (T / max(abs(l)))

print(T)
print(LA.eigvals(T))
print(max(LA.svdvals(T)))

# setting the integration time and time-step:
dt = 0.0001
tmax = 100 #input("How long (in seconds) should the system run? ")
tmax = float(tmax)
tlist = np.arange(0, tmax, dt)
nt = tlist.size

# setting the node variable (firing rates): 
#X = np.full((N, nt), 0.0)
X = np.zeros((N,nt))
X_intr = np.zeros((N,nt))


# initial perturbation:
#X[:,0] = np.random.random(N)
#X[0,0] = 0.1
#print(l)
#print(LA.svdvals(T))


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
D1 = np.zeros((N,nt))
D2 = np.zeros((N,nt))
L = 2000 # Input's length
D1[3,0:L] = 0.03 #* np.sin(5*tlist[0:L]) 
D1[4,0:L] = 0.15 * np.sin(40 * tlist[0:L])
D2[4,0:L] = 0.02 * np.sin(25 * tlist[0:L] + 1)
D = D1 + D2


# initiating the vector field:
F = np.zeros((N,nt)) 
F1 = np.zeros((N,nt))
F2 = np.zeros((N,nt))

def sgm(x):	
	return 1 / (1 + np.exp(-x))

print("Integration Started...")
for i in range(1,nt): # Heun's Method
	F[:,i-1] = sgm(T @ X[:,i-1] + D[:,i-1]) 
	X_intr[:,i] = X[:,i-1] + dt * F[:,i-1]  
	F[:,i] = sgm(T @ X_intr[:,i] + D[:,i])
	X[:,i] = X[:,i-1] + dt * (F[:,i-1] + F[:,i]) / 2

# storing the results:
# import pandas as pd
# pd.dataframe

	
# visualizations:
from matplotlib import pyplot as plt 
#plt.subplot(1,1)

plt.xlabel("time (sec)")
plt.ylabel("Nodes Responses")

plt.subplot(2,1,1)
plt.title("Input Signal")
for i in range(N):
	plt.plot(tlist, D[i,:], label = f'{i}')

plt.subplot(2,1,2)
plt.title("Response")
for i in range(N):
	plt.plot(tlist, X[i,:], label = f'{i}')

	#plt.legend(loc = 'lower right')
#plt.subplot(2,1)

plt.show()