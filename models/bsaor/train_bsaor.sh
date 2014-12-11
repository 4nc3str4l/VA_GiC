#!/usr/bin/env sh

/usr/local/caffe/tools/caffe train --solver=models/bsaor/bsaor_solver.prototxt 2> models/bsaor/output.txt
