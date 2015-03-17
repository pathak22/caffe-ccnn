from __future__ import division
import numpy as np
import os
import Image
from datetime import datetime
import os

fnames_val = {'pascal': np.loadtxt('/home/jonlong/caffe/pascal/seg11val.txt', str),
              'nyud': np.loadtxt('/home/jonlong/caffe/nyud/segval.txt', str),
              'nyudtest': np.loadtxt('/home/jonlong/caffe/nyud/segtest.txt', str),
              'sift-flow': np.loadtxt('/home/jonlong/caffe/sift-flow/segtest.txt', str)}

def prepare():
    name = os.path.basename(os.getcwd())
    save_dir = '/home/jonlong/x/caffe/exp/' + name
    if os.path.exists(save_dir):
        print save_dir, 'exists, bailing'
        exit(-1)
    os.mkdir(save_dir)
    print '>>> Starting', name
    return (os.path.join(save_dir, name + '_iter_{}/'),
            os.path.join(save_dir, name))

def compute_hist(net, save_dir, dataset):
    n_cl = net.blobs['upscore'].channels
    os.mkdir(save_dir)
    hist = np.zeros((n_cl, n_cl))
    for fname in fnames_val[dataset]:
        net.forward()
        h, _, _ = np.histogram2d(net.blobs['gt'].data[0, 0].flatten(),
                net.blobs['upscore'].data[0].argmax(0).flatten(),
                bins=n_cl, range=[[0, n_cl], [0, n_cl]])
        hist += h
        im = Image.fromarray(net.blobs['upscore'].data[0].argmax(0).astype(np.uint8), mode='P')
        im.save(os.path.join(save_dir, fname + '.png'))
    return hist

def seg_tests(solver, save_format, dataset):
    print '>>>', datetime.now(), 'Begin seg tests'
    solver.net.set_phase_test()
    solver.test_nets[0].share_with(solver.net)
    n_cl = solver.test_nets[0].blobs['upscore'].channels
    hist = compute_hist(solver.test_nets[0], save_format.format(solver.iter),
            dataset)
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'Iteration', solver.iter, 'overall accuracy', acc
    # per-class accuracy
    acc = np.zeros(n_cl)
    for i in range(n_cl):
        acc[i] = hist[i, i] / hist[i].sum()
    print '>>>', datetime.now(), 'Iteration', solver.iter, 'mean accuracy', np.nanmean(acc)
    # per-class IU
    iu = np.zeros(n_cl)
    for i in range(n_cl):
        iu[i] = hist[i, i] / (hist[i].sum() + hist[:, i].sum() - hist[i, i])
    print '>>>', datetime.now(), 'Iteration', solver.iter, 'mean IU', np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    print '>>>', datetime.now(), 'Iteration', solver.iter, 'fwavacc', \
            (freq[freq > 0] * iu[freq > 0]).sum()
    solver.net.set_phase_train()

def inst_tests(solver, save_format):
    print '>>>', datetime.now(), 'Begin inst tests'
    solver.net.set_phase_test()
    solver.test_nets[0].share_with(solver.net)
    net = solver.test_nets[0]
    save_dir = save_format.format(solver.iter)
    os.mkdir(save_dir)
    n = 0
    total_recall = 0
    total_iu = 0
    n50 = 0
    n70 = 0
    total_pixels = 0
    correct_pixels = 0
    for fname in fnames_val:
        net.forward()
        im = Image.fromarray(net.blobs['upscore'].data[0].argmax(0)
                .astype(np.uint8), mode='P')
        im.save(os.path.join(save_dir, fname + '.png'))
        gt_inds = np.unique(net.blobs['pgt'].data).astype(int)
        if gt_inds[0] == -1:
            gt_inds = gt_inds[1:]
        if gt_inds[0] == 0:
            gt_inds = gt_inds[1:]
        n += len(gt_inds)
        h, _, _ = np.histogram2d(net.blobs['pgt'].data[0, 0].flatten(),
                net.blobs['upscore'].data[0].argmax(0).flatten(),
                bins=101, range=[[0, 101], [0, 101]])
        rec = np.diag(h)[gt_inds] / h[gt_inds].sum(1)
        total_recall += rec.sum()
        iu = np.diag(h)[gt_inds]
        iu /= (h[gt_inds].sum(1) + h[:, gt_inds].sum(0) - iu)
        total_iu += iu.sum()
        n50 += (iu > 0.5).sum()
        n70 += (iu > 0.7).sum()
        total_pixels += h.sum()
        correct_pixels += np.diag(h).sum()
    print '>>>', datetime.now(), 'Iteration', solver.iter, 'mean recall', total_recall / n
    print '>>>', datetime.now(), 'Iteration', solver.iter, 'mean IU', total_iu / n
    print '>>>', datetime.now(), 'Iteration', solver.iter, 'portion > 0.5', n50 / n
    print '>>>', datetime.now(), 'Iteration', solver.iter, 'portion > 0.7', n70 / n
    print '>>>', datetime.now(), 'Iteration', solver.iter, 'overall accuracy', correct_pixels / total_pixels
    solver.net.set_phase_train()

def conn_tests(solver, save_format):
    print '>>>', datetime.now(), 'Begin conn tests'
    solver.net.set_phase_test()
    solver.test_nets[0].share_with(solver.net)
    net = solver.test_nets[0]
    save_dir = save_format.format(solver.iter)
    os.mkdir(save_dir)
    hist = np.zeros((2, 2))
    for fname in fnames_val:
        net.forward()
        im = Image.fromarray(net.blobs['left-conn'].data[0].argmax(0)
                .astype(np.uint8), mode='P')
        im.save(os.path.join(save_dir, fname + '-L.png'))
        im = Image.fromarray(net.blobs['top-conn'].data[0].argmax(0)
                .astype(np.uint8), mode='P')
        im.save(os.path.join(save_dir, fname + '-T.png'))
        h, _ , _ = np.histogram2d(net.blobs['left-gt'].data[0, 0].flatten(),
                net.blobs['left-conn'].data[0].argmax(0).flatten(),
                bins = 2, range=[[0, 2], [0, 2]])
        hist += h
        h, _ , _ = np.histogram2d(net.blobs['top-gt'].data[0, 0].flatten(),
                net.blobs['top-conn'].data[0].argmax(0).flatten(),
                bins = 2, range=[[0, 2], [0, 2]])
        hist += h
    print '>>>', datetime.now(), 'Iteration', solver.iter, 'overall accuracy', \
        np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'Iteration', solver.iter, 'total IU', \
            hist[1, 1] / (hist[0, 1] + hist[1, 0] + hist[1, 1])
    solver.net.set_phase_train()

def bound_tests(solver, save_format):
    print '>>>', datetime.now(), 'Begin bound tests'
    solver.net.set_phase_test()
    solver.test_nets[0].share_with(solver.net)
    net = solver.test_nets[0]
    save_dir = save_format.format(solver.iter)
    os.mkdir(save_dir)
    hist = np.zeros((2, 2))
    for fname in fnames_val:
        net.forward()
        im = Image.fromarray(net.blobs['cscore'].data[0].argmax(0)
                .astype(np.uint8), mode='P')
        im.save(os.path.join(save_dir, fname + '.png'))
        h, _ , _ = np.histogram2d(net.blobs['gt'].data[0, 0].flatten(),
                net.blobs['cscore'].data[0].argmax(0).flatten(),
                bins=2, range=[[0, 2], [0, 2]])
        hist += h
    print '>>>', datetime.now(), 'Iteration', solver.iter, 'overall accuracy', \
        np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'Iteration', solver.iter, 'total IU', \
            hist[1, 1] / (hist[0, 1] + hist[1, 0] + hist[1, 1])
    r = hist[1, 1] / hist[1].sum()
    p = hist[1, 1] / hist[:, 1].sum()
    print '>>>', datetime.now(), 'Iteration', solver.iter, 'recall', r
    print '>>>', datetime.now(), 'Iteration', solver.iter, 'precision', p
    print '>>>', datetime.now(), 'Iteration', solver.iter, 'F1', 2*r*p/(r + p)
    solver.net.set_phase_train()
