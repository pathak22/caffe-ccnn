import sys
sys.path.append('../../python')

import caffe
from caffe import surgery, score

import numpy as np
import os

weights = sys.argv[1]

caffe.set_phase_train()
caffe.set_mode_gpu()
caffe.set_device(1)

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

score.boundary_eval(solver, 'val', layer='prob')
