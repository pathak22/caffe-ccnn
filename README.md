# caffe-ccnn
=============

This is the branch of caffe created for [CCNN](https://github.com/pathak22/ccnn) project.
This is a pre-release Caffe branch.
This includes several unmerged PRs and has no guarantees.

Everything here is subject to change, including the history of this branch.
See `future.sh` for details.

Some Key Functionalities:

- Test Time DenseCRF Layer originally described in this [paper](http://graphics.stanford.edu/projects/densecrf/).
- Atrous or 'hole' algorithm described in the [arxiv paper](http://arxiv.org/abs/1412.7062). **Disclaimer**: The Atrous code in this repository is obtained (and modified) from the original implementation, released [here](https://bitbucket.org/deeplab/deeplab-public/).
- A rough version of MIL based Softmax Loss Layer for weakly supervised tasks.
- Background Weighted SoftmaxLossLayer.
- Some additional scripts in the folder ./my_script/
