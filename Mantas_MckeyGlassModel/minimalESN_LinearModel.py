# -*- coding: utf-8 -*-
"""
A minimalistic Echo State Networks demo with Mackey-Glass (delay 17) data 
in "plain" scientific Python.
by Mantas Lukoševičius 2012-2018
http://mantas.info
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA

# load the data
trainLen = 900
testLen = 1000
initLen = 100
# data = np.loadtxt('MackeyGlass_t17.txt')
data = np.loadtxt('Sine.txt')
print(np.min(data), np.max(data))
print(np.min(np.abs(data)), np.max(np.abs(data)))

# plot some of it
# plt.figure(10).clear()
plt.plot(data[0:1000])
plt.title('A sample of data')
# plt.show()


# generate the ESN reservoir
inSize = outSize = 1
resSize = 100
a = 1 # leaking rate
np.random.seed(42)
Win = (np.random.rand(resSize,inSize)-0.5) * 1
W = np.random.rand(resSize,resSize)-0.5 
# normalizing and setting spectral radius (correct, slow):
print('Computing spectral radius...'),
rhoW = np.max(np.abs(LA.eig(W)[0]))
print('done.')
W *= 0.98 / rhoW

# np.save(f'ReservoirSize{resSize}.npy', W)
# W = np.load('ReservoirSize3200.npy')

# allocated memory for the design (collected states) matrix
X = np.zeros((resSize,trainLen-initLen))
# set the corresponding target matrix directly
Yt = data[None,initLen+1:trainLen+1] #Arash: passing None as a first argument creates a 2 dim (1 x _) shape for Yt 

# run the reservoir with the data and collect X
x = np.zeros((resSize,1))
for t in range(trainLen):
    u = data[t]
    # x = (1-a)*x + a*( np.dot( Win, u ) + np.dot( W, x ) )
    x = np.dot( Win, u ) + np.dot( W, x ) 
    if t >= initLen:
        X[:,t-initLen] = x[:,0] #np.vstack((1,u,x))[:,0]    
# train the output by ridge regression
reg = 1e-8  # regularization coefficient
X_T = X.T
Yt = (1/2) * np.log((1 + Yt)/(1 - Yt)) ### applying nonlinearity here in the read-out func.
# Yt = (1/2) * ( np.log1p(Yt)/np.log1p(-Yt))
Wout = np.dot( np.dot(Yt,X_T), LA.inv( np.dot(X,X_T) + \
    reg*np.eye(resSize) ) )

# run the trained ESN in a generative mode. no need to initialize here, 
# because x is initialized with training data and we continue from there.
Y = np.zeros((outSize,testLen))
u = data[trainLen]
for t in range(testLen):
    x = (1-a)*x + a*( np.dot( Win, u ) + np.dot( W, x ) )
    y = np.tanh(np.dot( Wout, x ))
    Y[:,t] = y
    # generative mode:
    u = y
    ## this would be a predictive mode:
    #u = data[trainLen+t+1] 

# compute MSE for the first errorLen time steps
errorLen = testLen
mse = np.sum( np.square( data[trainLen+1:trainLen+errorLen+1] - 
    Y[0,0:errorLen] ) ) / errorLen
print('MSE = ' + str( mse ))
    
# plot some signals
plt.subplot(2,1,1)
# plt.figure(1).clear()
# plt.xlim(0, 400)
plt.plot( data[trainLen+1:trainLen+testLen+1], 'g' )
plt.plot( Y.T, 'b' )
plt.title('Target and generated signals $y(n)$ starting at $n=0$')
plt.legend(['Target signal', 'Free-running predicted signal'])



# plt.subplot(2,1,2)
# # plt.figure(2).clear()
# plt.plot( X[0:20,0:trainLen].T )
# plt.title('Some reservoir activations $\mathbf{x}(n)$')


# plt.subplot(3,1,3)
# # plt.figure(3).clear()
# # plt.bar( range(1+inSize+resSize), Wout.T )
# plt.title('Output weights $\mathbf{W}^{out}$')

plt.show()

print("size of range(1+inSize+resSize)", 1+inSize+resSize)
print((Wout.T).shape)