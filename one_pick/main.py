'''
Peaking one best peak that differentiate class N vs class M

- Loss Function


- Optimisation Method

'''

import sys
import os

sys.path.append(os.path.abspath('../tools'))
import data_prep

sys.path.append(os.path.abspath('../one_pick'))
import PSO

# Get data
_, data1 = data_prep.get_MNIST(10, 2)
_, data2 = data_prep.get_MNIST(10, 4)

### Run PSO
pso = PSO.PSO()
# k= parent size, swarm_size=number of parameter set in one iteration
result = pso.run([2, 4], [data1, data2], swarm_size=5,options={'c1': 1.5, 'c2':1.5, 'w':0.5, 'k':2, 'p':1}, iters=5)
pso.test([2, 4], [data1, data2], result)

###

