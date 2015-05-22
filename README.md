# caffe-pathak : fcn-mil branch
==============================

This is the fcn-mil branch. It is developed over fcn branch of Evan's Caffe.
This is a pre-release Caffe branch for fully convolutional networks.
This includes several unmerged PRs and has no guarantees.

Everything here is subject to change, including the history of this branch.

See `future.sh` for details.

Additional Functionalitites over Master :

- My version of EM Adapt MIL Loss Layer
- Background Weighted SoftmaxLossLayer
- Test Time DenseCRF Layer Added (Philipp NIPS '11)
- Permutohedral Filter Layer (DenseCRF)

Included Philipp's Functionalities over Master :

- Running Average Layer (Remains same across run: Philipp)
- Statitical Loss Layer (Compute IOU : Only in Python Layer : Philipp) 
- Temp Memory Layer (Philipp)
- Repack Layer (Philipp)
- Solver as string in python layer (Philipp)
- Base Convolution Layer Modified (Philipp) 
- Repacking Layer for denser stride network (Philipp)

Consider adding :

- https://github.com/BVLC/caffe/pull/1977 for saving memory
- https://github.com/BVLC/caffe/pull/1733 for pycaffe net specification
- https://github.com/BVLC/caffe/pull/2016 for saving memory during test time
