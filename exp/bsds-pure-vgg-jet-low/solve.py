import sys
sys.path.append('../../python')

import caffe
from caffe import surgery, score

import numpy as np
import os

#weights = '/home/shelhamer/dpb/exp/vgg16fc.caffemodel'
weights = '/home/shelhamer/dpb/exp/pure-scale/train_iter_27175.caffemodel'

caffe.set_phase_train()
caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

for _ in range(6):
    print 'epoch {}'.format(_)
    solver.step(1087*5)
    score.boundary_eval(solver, ('bsds', 'val'), layer='prob')
