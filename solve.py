import sys
sys.path.append('../../python')

import caffe
from caffe import surgery, score

import numpy as np
import os
import cPickle as pickle

save_format = os.getcwd() + '/out_{}'

# base net
base = 'base.prototxt'
weights = 'vgg16fc.caffemodel'
base_net = caffe.Net(base, weights, caffe.TEST)

# init
caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.SGDSolver('solver.prototxt')

# surgeries
surgery.transplant(solver.net, base_net)

interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# load IDs of the TEST phase data
val = np.loadtxt('list.txt', dtype=str)

for _ in range(100):
    solver.step(1000)
    score.seg_tests(solver, save_format, val, layer='score', gt='label')
