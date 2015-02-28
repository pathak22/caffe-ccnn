# to plot loss call with the name of the logfile.

from pylab import *
from sys import argv
from scipy.ndimage import filters
import re

if len(argv) < 2:
    print('Usage: %s logfile' %argv[0])
    exit(1)

loss = []
iter = []
additional_loss = {}
lines = open(argv[1],'r').read()
r = re.compile('Train net output .* (.*) = .* = (.*) loss\)')
for l in lines.split('\n'):
    s = l.split()
    losses = r.findall(l)
    if losses:
        if not losses[0][0] in additional_loss:
            additional_loss[losses[0][0]] = []
        additional_loss[losses[0][0]].append(float(losses[0][1]))
    
    if len(s)>4 and  s[-2] == '=' and s[-3] == 'loss':
        loss.append(float(s[-1]))
        iter.append(float(s[-4][:-1]))
plot(iter, loss, label='loss')
plot(iter, filters.uniform_filter1d(loss, 100), label='smooth loss')
for i in additional_loss:
    plot(iter[:len(additional_loss[i])], additional_loss[i], label=i)
legend()
show()
