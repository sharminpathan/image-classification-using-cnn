import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
import glob
import caffe

MODEL_FILE = '/home/sharmin/Desktop/alexNet/deploy.prototxt'
PRETRAINED = '/home/sharmin/Desktop/snapshots/snapshots__iter_200.caffemodel'
BINARY_PROTO_MEAN_FILE = "/home/sharmin/Desktop/mean.binaryproto"

blob = caffe.proto.caffe_pb2.BlobProto()
data = open(BINARY_PROTO_MEAN_FILE, 'rb').read()
blob.ParseFromString(data)
mean_arr = np.array(caffe.io.blobproto_to_array(blob))[0]

caffe.set_mode_cpu()

net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=mean_arr.mean(1).mean(1),channel_swap=(2,1,0),raw_scale=255,image_dims=(32,32))

f=open("output.csv","w")
f.write("filename,c_0,c_1,c_2,c_3,c_4,c_5,c_6,c_7,c_8,c_9\n")

outf = open("/home/sharmin/Desktop/outputPredict.txt","w")

number_of_files_processed=0
for file in glob.glob("/home/sharmin/Desktop/testImages/*"):
	number_of_files_processed += 1
	FileName = file.split("/")[-1]
	input_image = caffe.io.load_image(file)
	prediction = net.predict([input_image])
	s = FileName+","
	for probability in prediction[0]:
		s+=str(probability)+","
	s = s[:-1]+"\n"
	f.write(s)
	p = prediction[0].argmax()
	outf.write(FileName+" "+str(p)+"\n")
	#print ("Number of files : "+str(number_of_files_processed))
    #print ("predicted class:"+str(prediction[0].argmax()))
    #print "**********************************************"
