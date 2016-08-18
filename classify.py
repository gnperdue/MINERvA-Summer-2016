#classify all events in testing .hdf5, dump predictions to .txt

import sys
sys.path.insert(0, '/opt/caffe/python')
import caffe
import numpy as np

#Example usage:
#python classify.py <path to prototxt> <path to model> <test iter> <test batch size> <output file>

#Run parameters
if len(sys.argv) == 6:
    prototxt = sys.argv[1]
    model = sys.argv[2]
    iter = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    outfile = sys.argv[5]
else:
    sys.exit("Incorrect number of arguments")


# Initialize Network
caffe.set_mode_cpu()
# Import net
net = caffe.Net(prototxt, model, caffe.TEST)

with open(outfile, "w") as f:
    for i in xrange(test_iter):
        net.forward()
        predictions = net.blobs['output'].data.argmax(1)
        predvecs = net.blobs['output'].data
        eventids = net.blobs['eventids'].data
        for j in xrange(test_batch_size):
            prediction = predictions[j]
            predvec = predvec[j]
            eventid = eventids[j]
            f.write(eventid+"\t"+prediction+"\t"+predvec+"\n")
f.close()




