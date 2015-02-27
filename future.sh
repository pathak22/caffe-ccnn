#!/bin/bash
git checkout master
git branch -D future
git checkout -b future
## merge PRs
# deconv layer, coord maps, net pointer, crop layer
hub merge https://github.com/BVLC/caffe/pull/1976
# gradient accumulation
hub merge https://github.com/BVLC/caffe/pull/1977
## commit
git add future.sh
git add README.md
git commit -m 'add README + creation script'
