from __future__ import division
import numpy as np
import os
from datetime import datetime
import os

def compute_hist(outs, truths, n_cl):
    hist = np.zeros((n_cl, n_cl))
    for idx in outs:
        h, _, _ = np.histogram2d(truths[idx].flatten(),
                outs[idx].flatten(),
                bins=n_cl, range=[[0, n_cl], [0, n_cl]])
        hist += h
    return hist

def seg_tests(outs, truths, n_cl):
    print '>>>', datetime.now(), 'Begin seg tests'
    hist = compute_hist(outs, truths, n_cl)
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'overall accuracy', acc
    # per-class accuracy
    acc = np.zeros(n_cl)
    for i in range(n_cl):
        acc[i] = hist[i, i] / hist[i].sum()
    print '>>>', datetime.now(), 'mean accuracy', acc.mean()
    # per-class IU
    iu = np.zeros(n_cl)
    for i in range(n_cl):
        iu[i] = hist[i, i] / (hist[i].sum() + hist[:, i].sum() - hist[i, i])
    print '>>>', datetime.now(), 'mean IU', iu.mean()
    freq = hist.sum(1) / hist.sum()
    print '>>>', datetime.now(), 'fwavacc', np.dot(freq, iu)
