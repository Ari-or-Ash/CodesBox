
'''generating a reservoir (adjacency matrix)'''

import numpy as np
from scipy import linalg as LA


N = 10 # nodes number

creating a random connectivity matrix:
# W = np.random.random((N,N))
W = np.random.uniform(-1 ,1 ,(N,N))

# making the matrix sparse: 
s = 0.98	# (rough)sparcity parameter = (1 - density)

S = int(s * N)
for i in range(N):
	for j in range(S):
		m = np.random.randint(0,N)
		W[i,m] = 0.0

	
# ensuring the negativity of eigenvalues (statibility of the FPs):

# Diagonal = -(np.sqrt(1 - s)) * N - 1
Diagonal = -(np.sqrt(1 - s)) * N - 1
np.fill_diagonal(W, Diagonal)

print(W)

# ensuring the connectedness of the whole network:
Test = np.copy(W)
Test[np.where(Test != 0)] = 1 
Lpl = Test.T + Test
np.fill_diagonal(Lpl, 0)
Lpl[np.where(Lpl != 0)] = -1
degrees = -np.sum(Lpl, axis=1)
np.fill_diagonal(Lpl, degrees)
EVs = LA.eigvals(Lpl)
zeros = np.isclose(EVs, 0.0)
components_num = zeros[np.where(zeros==True)].size
print("number of the graph components:", components_num)


# ensuring the existence of cycles:
## ... coming soon


# ensuring ESP?! How?
### ESP condition: max(LA.svdvals(W)) < 1
#print(LA.svdvals(W))

if components_num == 1:
	
	l = LA.eigvals(W)
	W = 0.85 * (W / max(abs(l)))
	'''
	l, v = LA.eig(W)
	W = 0.97 * (W / max(abs(l)))

	#l, v = LA.eig(W)
	#print(l, v)
	'''
	l = LA.eigvals(W)
	SVs = LA.svdvals(W)
	max_SV = np.max(SVs)
	max_l = np.max(l)
	print(W)
	print(max_l)
	print(max_SV)

	if max_l < 0 and max_SV < 1:
		np.save(f'AdjacencyMatrix_N={N}_max(SV)={max_SV}.npy', W)
	else:
		print("the generated matrix didn't satisfy all the necessary conditions. try again!")

else:
	print("the generated matrix didn't satisfy all the necessary conditions. try again!")



