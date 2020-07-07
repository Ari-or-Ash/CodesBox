import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 10, 0.001)
fr = 7
Sine = np.sin(fr*t + np.pi/5)

np.savetxt(f'Sine_fr={fr}', Sine)
# plt.plot(t, Sine)
# plt.show()