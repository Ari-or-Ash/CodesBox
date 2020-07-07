
'''generating a reservoir (adjacency matrix)'''

import numpy as np
from scipy import linalg as LA


N = 100 # nodes number

# creating a random connectivity matrix:

W = np.random.uniform(-1,1,(N,N))
s = 0.6 # sparsity score
for i in range(N):
	M = np.random.choice(N, round(s*N), replace=False)
	for j in M:
		W[i,j] = 0.0
	
# ensuring the negativity of eigenvalues (statibility of the FPs):
Diagonal = -(np.sqrt(1 - s)) * N #- 1
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


# ensuring ESP with: max(LA.svdvals(W)) < 1
l = LA.eigvals(W)
W = 0.85 * (W / max(abs(l)))
l = LA.eigvals(W)
l_max = np.max(l)
SVs = LA.svdvals(W)
max_SV = np.max(SVs)
print(l_max)
print(max_SV)


if components_num == 1:

	if l_max < 0 and max_SV < 1:
		np.save(f'AdjacencyMatrix_V2_N={N}_sparsity={s}_max(SV)={"%.5f" % max_SV}.npy', W)
	else:
		print("the generated matrix didn't satisfy either of the stability & the ESP conditions. try again!")

else:
	print("the generated matrix isn't one single component. try again!")
