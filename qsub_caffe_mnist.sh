#!/bin/bash
DAT=`date +%s`
OUTFILENAME="./caffe_mnist_${DAT}.txt"
JOBNAME="caffe-test-${DAT}"
qsub -o $OUTFILENAME job_caffe_mnist.sh -N $JOBNAME
