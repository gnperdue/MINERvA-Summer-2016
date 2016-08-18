#produce confusion matrix

import sys
sys.path.insert(0, '/opt/caffe/python')
import caffe
import numpy as np

#Example usage:
#python classify.py <n-hadmultmeas/segments> <path to prototxt> <path to model> <test iter> <test batch size> 

#Run parameters
if len(sys.argv) == 6: 
    label = sys.argv[1]
    prototxt = sys.argv[2]
    model = sys.argv[3]
    iter = int(sys.argv[4])
    batch_size = int(sys.argv[5])
else:
    sys.exit("Incorrect number of arguments")



if label=="segments":
    confusionMatrix = np.zeros([11,11])
else:
    confusionMatrix = np.zeros([6,6]) #change to 6x6 for hadmult


# Initialize Network
caffe.set_mode_cpu()
# Import net
net = caffe.Net(prototxt, model, caffe.TEST)

for i in xrange(iter):
    net.forward()
    predictions = net.blobs['output'].data.argmax(1)
    actuals = net.blobs[label].data
    for j in xrange(batch_size):
        confusionMatrix[predictions[j],actuals[j]] +=1

print confusionMatrix




