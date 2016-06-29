#!/bin/bash
#PBS -S /bin/bash
#PBS -N caffe-hadmult_epsilon
#PBS -j oe
DAT=`date +%s`
#PBS -o ./out_job_caffe_hadmult_epsilon_${DAT}.txt
#PBS -l nodes=1:gpu,walltime=24:00:00
#PBS -A minervaG
#PBS -q gpu
#restore to turn off email #PBS -m n

# print identifying info for this job
echo "Job ${PBS_JOBNAME} submitted from ${PBS_O_HOST} started "`date`" jobid ${PBS_JOBID}"

# these are broken?...
# nCores=$['cat ${PBS_COREFILE} | wc --lines']
# nNodes=$['cat ${PBS_NODEFILE} | wc --lines']
# echo "NODEFILE nNodes=$nNodes (nCores=$nCores):"

cat ${PBS_NODEFILE}

cd $HOME
source caffe_gpu_setup.sh

cd ${PBS_O_WORKDIR}
echo "PBS_O_WORKDIR is `pwd`"

# Always use fcp to stage any large input files from the cluster file server
# to your job's control worker node. All worker nodes have attached 
# disk storage in /scratch.

# There is no fcp on the gpu nodes...
# /usr/local/bin/fcp -c /usr/bin/rcp tevnfsp:/home/perdue/Datasets/mnist.pkl.gz /scratch
# ls /scratch

# Always use fcp to copy any large result files you want to keep back
# to the file server before exiting your script. The /scratch area on the
# workers is wiped clean between jobs.

LMDBDIR=/phihome/sohini/caffe/hadmult
NETWORKDIR=/phihome/sohini/caffe/hadmult

echo " &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& "
echo "                     Train                             "

caffe.bin train -solver $NETWORKDIR/hadmult_epsilon.solver

echo " &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& "
echo "                     Test                              "

caffe.bin test -model $NETWORKDIR/hadmult_epsilon.prototxt \
  -weights $NETWORKDIR/hadmult_epsilon_iter_150000.caffemodel -gpu 0

exit 0
