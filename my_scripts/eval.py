import sys
sys.path.append('../../python')

import caffe
from caffe import surgery, score

import numpy as np
import os

save_format = os.getcwd() + '/out_{}'

weights = sys.argv[1]

caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# load IDs of the TEST phase data
val = np.loadtxt('list.txt', dtype=str)

score.seg_tests(solver, save_format, val, layer='score', gt='label')
