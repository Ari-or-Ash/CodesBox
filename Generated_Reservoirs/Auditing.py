import numpy as np
from scipy import linalg as LA
import os

os.chdir('Generated_Reservoirs')
for file in os.listdir():
    f_name, f_ext = os.path.splitext(file)
    if f_ext == '.npy':
        W = np.load(file)
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
        if components_num == 1:
            l = LA.eigvals(W)
            SVs = LA.svdvals(W)
            SV_max = np.max(SVs)
            print("All the EVs have negative real parts? and the ESP guaranteed?", 0 > np.max(np.real(l)),",", 1 > SV_max)
        
        else:
            print("this graph was not a single component, try another one!")

		# # checking for existence of cycles
		# W2 = W @ W
		# W3 = W2 @ W
		# W4 = W3 @ W
    