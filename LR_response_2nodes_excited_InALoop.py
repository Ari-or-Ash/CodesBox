import time
t0 = time.time()
import numpy as np
from scipy import linalg as LA
import os
from matplotlib import pyplot as plt 
# import pandas as pd

'''
--------------------------------------------------------------------------------------------
|                 Simulating Reservoir's response to exciting two nodes                    |
|			       	with the phase shifted copies of the same input         	       	   |
|__________________________________________________________________________________________|	
'''

Res_Location = os.path.abspath('Generated_Reservoirs')
W = 1 * np.load(f'{Res_Location}\AdjacencyMatrix_N=10_max(SV)=0.933745180396509.npy')
# W = 1 * np.load(f'{Res_Location}\AdjacencyMatrix_V2_N=10_sparcity=0.9_max(SV)=0.9936.npy')
# W = 1 * np.load(f'{Res_Location}\AdjacencyMatrix_V2_N=10_sparcity=0.8_max(SV)=0.9279.npy')
# W = 1 * np.load(f'{Res_Location}\AdjacencyMatrix_V2_N=10_sparcity=0.7_max(SV)=0.92600.npy')
# W = 1 * np.load(f'{Res_Location}\AdjacencyMatrix_V2_N=10_sparcity=0.6_max(SV)=0.92832.npy')
# W = 1 * np.load(f'{Res_Location}\AdjacencyMatrix_V2_N=10_sparcity=0.5_max(SV)=0.91600.npy')
# W = 1 * np.load(f'{Res_Location}\AdjacencyMatrix_V2_N=100_sparsity=0.6_max(SV)=0.86964.npy')
N = W.shape[0]
SV_max = "%.5f" % np.max(LA.svdvals(W))
EVs = LA.eigvals(W)
# negative_inverse_tau = set(np.diagonal(W))
# print(negative_inverse_tau)

def Main(PhaseDif):

	# setting the integration time and time-step:
	dt = 0.001
	tmax = 25 #input("How long (in seconds) should the system run? ")
	tmax = float(tmax)
	tlist = np.arange(0, tmax, dt)
	nt = tlist.size

	# setting the node variable (firing rates): 

	print(f'N = {N}')

	X = np.zeros((N,nt))
	X_intr = np.zeros((N,nt))

	# initial perturbation:
	#X[:,0] = np.random.random(N)
	#X[0,0] = 0.1


	drived_node1 = 1 #int(input("which node to drive? "))
	drived_node2 = 7 #int(input("which other node to drive? "))
	# PhaseDif = float(input('Enter the PhaseDif> '))
	drived_nodes = [drived_node1, drived_node2] # will use this for plotting at the end
	# 1st signal
	A1 = np.zeros(N)
	#A1[3] = 0.03
	A1[drived_node1] = 0.15

	FR1 = np.zeros(N)
	FR1[drived_node1]= 1.0 #40.0

	PHI1 = np.zeros(N)
	#PHI1[3] = np.pi / 2
	PHI1[drived_node1] = 0.0 #1.571 #np.pi / 2


	# 2nd signal
	A2 = np.zeros(N)
	#A2[3] = 0.03
	A2[drived_node2] = A1[drived_node1] # 0.15

	FR2 = np.zeros(N)
	FR2[drived_node2]= FR1[drived_node1] #3.0 #40.0

	PHI2 = np.zeros(N)
	#PHI1[3] = np.pi / 2
	PHI2[drived_node2] = PHI1[drived_node1] + PhaseDif #np.pi / 2



	t1 = 0.0 # Input's start time
	t2 = 20.0 # Input's end time
	ti_1 = int(t1 / dt)
	ti_2 = int(t2 / dt)
	U = np.zeros((N,nt))

	for i in range(N):
		U [i,ti_1:ti_2] = A1[i] * np.sin(FR1[i] * tlist[0:ti_2 - ti_1] + PHI1[i]) + \
						A2[i] * np.sin(FR2[i] * tlist[0:ti_2 - ti_1] + PHI2[i])
	
	# initiating the vector field:

	# F = np.zeros((N,nt))
	# # print(np.max(EVs))

	

	# # Integrating the Dynamics (Harvesting Reservoir States)
	# print("Integration Started...")
	# t1 = time.time()
	# for i in range(1,nt): # Heun's Method
	# 	F[:,i-1] = W @ X[:,i-1] + U[:,i-1] 
	# 	X_intr[:,i] = X[:,i-1] + dt * F[:,i-1]  
	# 	F[:,i] = W @ X_intr[:,i] + U[:,i]
	# 	X[:,i] = X[:,i-1] + dt * (F[:,i-1] + F[:,i]) / 2
	# print("integration's duration in sec(s): ", time.time() - t1)
	# X = np.around(X, 7)

	x = np.zeros(N)
	x_intro = np.zeros(N)
	f_1 = np.zeros(N)
	f_2 = np.zeros(N)
	print((W @ x).shape)
	print(U[:,i-1].shape)

	print("Integration Started...")
	t1 = time.time()
	for i in range(1,nt): # Heun's Method
		f_1 = (W @ x) + U[:,i-1]#.reshape((10,1))
		x_intr = (x + dt * f_1)
		f_2 = (W @ x_intr) + U[:,i]#.reshape((10,1))
		x = x + dt * (f_1 + f_2) / 2
		X[:,i] = x
	print("integration's duration in sec(s): ", time.time() - t1)
	X = np.around(X, 7)

	## storing the results:
	my_path = os.path.abspath('checking_the_Phase_Effect')
	# time_stamps = {'time': tlist} 
	# responses = {f'#{i}_response': X[i,:].round(6) for i in range(N)}
	# data = {**time_stamps, **responses}
	# df = pd.DataFrame(data) 
	# df.to_csv(f'{my_path}/PhaseDif={"%.3f" % PhaseDif}_drived_nodes_#{drived_node1}_#{drived_node2}.csv')

	# np.save(f'{my_path}/MtID_{SV_max}_N={N}_PhaseDif={"%.3f" % PhaseDif}_drived_nodes_#{drived_node1}_#{drived_node2}.npy', X)

	# visualizations:
	plt.figure(figsize=[14,6.5])
	plt.xlabel("time (sec)")
	plt.ylabel("Nodes Responses")
	plt.title(f'Response of a Linear Reservoir of {N} nodes to driving nodes #{drived_node1} & {drived_node2}\n with the Input signals of Amplitudes {float(A1[np.where(A1!=0)])}, frequencies {float(FR1[np.where(FR1!=0)])} Hz & PhaseDif={"%.3f" % PhaseDif}')
	plt.ylim(-0.17,0.17)
	for i in range(N):
		# plt.subplot(2,1,1)
		# plt.plot(tlist, U[i,:])
		# plt.subplot(2,1,2)
		plt.plot(tlist, X[i,:], label = f'$\lambda_{i}$: {"%.3f" % np.real(EVs[i])} {"%.3f" % np.imag(EVs[i])}i')

	for i in drived_nodes:	
		plt.plot(tlist, U[i,:], ':', color='k')
	plt.legend(fontsize='x-small',loc = 'upper right')


	# plt.savefig(f'{my_path}/MtID_{SV_max}_N={N}_Drived_Nodes_{drived_node1}&{drived_node2}_PhaseDif_{"%.3f" % PhaseDif}.jpg')
	# # plt.show()
	plt.close()


for i in np.arange(0, 6.4, 0.2):
	print("Phase Difference=", "%.3f" % i)
	Main(i)

time_taken = time.time() - t0
print("Simulation Time, in sec(s):", time_taken)