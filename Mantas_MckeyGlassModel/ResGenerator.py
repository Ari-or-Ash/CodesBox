
resSize = 1000

np.random.seed(42)
W = np.random.rand(resSize,resSize)-0.5 

# normalizing and setting spectral radius (correct, slow):
print('Computing spectral radius...'),
rhoW = np.max(np.abs(LA.eig(W)[0]))
print('done.')

W *= 1.25 / rhoW

np.save(f'ReservoirSize{resSize}.npy', W)