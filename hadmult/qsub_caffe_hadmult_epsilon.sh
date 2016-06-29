#!/bin/bash
DAT=`date +%s`
OUTFILENAME="./caffe_hadmult_epsilon_${DAT}.txt"
JOBNAME="caffe-test-${DAT}"
qsub -o $OUTFILENAME job_caffe_hadmult_epsilon.sh -N $JOBNAME