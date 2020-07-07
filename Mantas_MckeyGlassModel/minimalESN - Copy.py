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
trainLen = 2000
testLen = 500
initLen = 100
data = np.loadtxt('MackeyGlass_t17.txt')
print(data.shape)

# plot some of it
# plt.figure(10).clear()
plt.plot(data[0:1000])
plt.title('A sample of data')
# plt.show()


# generate the ESN reservoir
inSize = outSize = 1
resSize = 500
a = 1 # leaking rate
np.random.seed(42)
Win = (np.random.rand(resSize,inSize)-0.5) * 1
Win_ = (np.random.rand(resSize,inSize)-0.5) * 1
W = np.random.rand(resSize,resSize)-0.5 
# normalizing and setting spectral radius (correct, slow):
print('Computing spectral radius...'),
rhoW = np.max(np.abs(LA.eig(W)[0]))
print('done.')
W *= 1.25 / rhoW

# allocated memory for the design (collected states) matrix
X = np.zeros((resSize,trainLen - initLen))

# set the corresponding target matrix directly
Yt = data[None,initLen+1:trainLen+1] ###my Arash: passing None as a first argument creates a 2 dim (1 x _) shape for Yt 

# run the reservoir with the data and collect X
x = np.zeros((resSize, 1))
# print(x.shape)
# data_ext = np.hstack((data, np.zeros(500)))
# print(data_ext.shape)
# print(data_ext[9995:10010])
for t in range(trainLen):
    u = data[t]
    # u = data_ext[t]
    x = (1 - a) * x + a * np.tanh( np.dot(Win, u ) + np.dot( W, x ) )
    if t >= initLen:
        X[:,t-initLen] = x[:,0] #np.vstack((1,u,x))[:,0]

# print(np.vstack((1,u,x)).shape)
# print(np.vstack((1,u,x))[:,0].shape)

### my added block seeing if the code could be rewritten in accord with what i'm doing in my thesis
# X_ = np.zeros((resSize, trainLen))
# for t in range(trainLen):
#     u = data[t]
#     X_[:,t] = (1 - a) * X_[:,t] + a * np.tanh(np.dot(Win, np.array((1,u))) + np.dot(W, X_[:,t]))
# # for t in range(trainLen, 1000):
# #     X_[:,t] = (1 - a) * X_[:,t] + a * np.tanh(np.dot(Win, np.array((1,0))) + np.dot(W, X_[:,t]))

# # data_ = np.hstack((data, np.zeros(1000)))    
# X_New = np.vstack((np.ones(trainLen), data[:trainLen], X_))[:,initLen:]


# train the output by ridge regression
reg = 1e-8  # regularization coefficient
X_T = X.T
Wout = np.dot( np.dot(Yt,X_T), LA.inv( np.dot(X,X_T) + \
    reg*np.eye(resSize) ) )

print(Wout.shape)
###
# X__T = X_.T
# Wout_ = np.dot( np.dot(Yt,X__T), LA.inv( np.dot(X_,X__T) + \
#     reg*np.eye(1+inSize+resSize) ) )



# run the trained ESN in a generative mode. no need to initialize here, 
# because x is initialized with training data and we continue from there.
Y = np.zeros((outSize,testLen))
u = data[trainLen]
for t in range(testLen):
    x = (1-a)*x + a*np.tanh( np.dot( Win, u ) + np.dot( W, x ) )
    y = np.dot( Wout, x )
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
plt.plot( data[trainLen+1:trainLen+testLen+1], 'g' )
plt.plot( Y.T, 'b' )
plt.title('Target and generated signals $y(n)$ starting at $n=0$')
plt.legend(['Target signal', 'Free-running predicted signal'])



plt.subplot(2,1,2)
# plt.figure(2).clear()
plt.plot( X[0:20,:200].T )
plt.title('Some reservoir activations $\mathbf{x}(n)$')

# plt.plot( X_New[0:20,0:500].T )
# plt.title('Some reservoir activations $\mathbf{x}(n)$')


# plt.subplot(3,1,3)
# # plt.figure(3).clear()
# # plt.bar( range(1+inSize+resSize), Wout.T )
# plt.title('Output weights $\mathbf{W}^{out}$')

plt.show()

