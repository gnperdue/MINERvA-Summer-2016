#classify all events in testing .hdf5, dump predictions to .txt
#use for noDANN

import sys
sys.path.insert(0, '/opt/caffe/python')
import caffe
# import numpy as np

# Example usage:
# python classify.py <path to prototxt> <path to model> <test iter> \
#   <test batch size> <label:n-hadmultmeas/segments> <output file>

# Run parameters
if len(sys.argv) == 7:
    prototxt = sys.argv[1]
    model = sys.argv[2]
    iter = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    label = sys.argv[5]
    outfile = sys.argv[6]
else:
    sys.exit("Incorrect number of arguments")
    
    
# Initialize Network
caffe.set_mode_cpu()
# Import net
net = caffe.Net(prototxt, model, caffe.TEST)

with open(outfile, "w") as f:
    for i in xrange(iter):
        net.forward()
        predictions = net.blobs['output'].data.argmax(1)
        actuals = net.blobs[label].data
        predvecs = net.blobs['output'].data
        eventids = net.blobs['eventids'].data
        for j in xrange(batch_size):
            actual = actuals[j]
            prediction = predictions[j]
            predvec = predvecs[j]
            eventid = eventids[j]
            line = str(eventid) + "\t" + str(prediction) +"\t" + str(actual) + "\t" + str(predvec)+"\n"
            print line
            f.write(line)
f.close()




