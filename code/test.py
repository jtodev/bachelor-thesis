import qutip as qtp
import numpy as np


a = np.random.normal(0,1,(4,3))
print(a)

quant = qtp.Qobj(a)
print(quant)