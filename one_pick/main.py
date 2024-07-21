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

num = 2
con = 4
# Get data
_, data1 = data_prep.get_MNIST(100, num)
_, data2 = data_prep.get_MNIST(100, con)

### Run PSO
pso = PSO.PSO()
# k= parent size, swarm_size=number of parameter set in one iteration
result = pso.run([num, con], [data1, data2], swarm_size=5,options={'c1': 1.5, 'c2':1.5, 'w':0.5, 'k':2, 'p':1}, iters=5)

_, data1 = data_prep.get_MNIST(200, num)
_, data2 = data_prep.get_MNIST(200, con)
pso.test([num, con], [data1, data2], result)

###

