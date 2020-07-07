"""
defining a XOR truth-table (function) based on phase differences(delta) and amplitude ratios(ro),
then generatig a sequence of desired length for training our Reservoir.
"""
import numpy as np
# delta = np.random.uniform(2*np.pi)
# ro = np.random.uniform(1)

def f_out(Delta, Ro ):
	if (Delta < np.pi and Ro < 1/2 ) or (Delta >= np.pi  and Ro >= 1/2 ):
		return 0
	if (Delta < np.pi and Ro >= 1/2 ) or (Delta >= np.pi and Ro < 1/2 ):
		return 1

## testing if the function work correctly and generating an (input(tuple), output) sequence
# for i in range(10):
# 	delta = np.random.uniform(0, 2*np.pi)
# 	ro = np.random.uniform(0, 1)
# 	print(f"{delta} XOR {ro} = ", f_out(delta, ro))

seq_length = 20	

delta = np.random.uniform(0, 2*np.pi, seq_length)
ro = np.random.uniform(0, 1, seq_length)

XOR = np.zeros(seq_length)
for i in range (seq_length):
	XOR[i] = f_out(delta[i], ro[i])

delta = delta.reshape(seq_length, 1)
ro = ro.reshape(seq_length, 1)
XOR = XOR.reshape(seq_length, 1)

Input = np.hstack((delta, ro))
Dict = np.hstack((Input, XOR))


# XOR = ( f_out(delta[i], ro[i]) for i in range(seq_length) )
# Inputs = zip (delta, ro)
# Dict = zip (Inputs, XOR)

np.savetxt('XOR_Sequence.txt', Dict)

for i in Dict:
	print(i)