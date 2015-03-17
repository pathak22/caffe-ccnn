from __future__ import division
import caffe
from caffe import tools
import test
from util import Timer
import numpy as np
import os

save_format, snapshot_prefix = test.prepare()

weights = '/home/jonlong/x/caffe/models/VGG/VGG16_full_conv.caffemodel'

caffe.set_device(6)
solver = caffe.SGDSolver('solver.prototxt')
solver.set_snapshot_prefix(snapshot_prefix)
solver.net.set_phase_train()
solver.net.set_mode_gpu()
solver.net.copy_from(weights)

for _ in range(1000):
    solver.step(200)
    test.seg_tests(solver, save_format, 'pascal')
