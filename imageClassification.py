import urllib2
import cStringIO
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from PIL import Image
from pyspark import SparkContext
from pyspark.sql import SparkSession
from scipy import misc

np.set_printoptions(threshold=np.inf)
def average(pixel):
	return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

data = urllib2.urlopen("https://s3.amazonaws.com/eds-uga-csci8360/data/project3/metadata/X_small_train.txt")
label = urllib2.urlopen("https://s3.amazonaws.com/eds-uga-csci8360/data/project3/metadata/y_small_train.txt")

fileData = open('/home/sharmin/Desktop/data.txt','w')
for d1 in data:
	fileData.write(str(d1[0:5])+"\n")
fileLabel = open('/home/sharmin/Desktop/label.txt','w')
for l1 in label:
	fileLabel.write(str(l1))

fileData.close()
fileLabel.close()

with open('/home/sharmin/Desktop/path_label','a') as res,open('/home/sharmin/Desktop/data.txt','r') as d1, open('/home/sharmin/Desktop/label.txt','r') as l1:
	for line1,line2 in zip(d1,l1):
		res.write("https://s3.amazonaws.com/eds-uga-csci8360/data/project3/images/"+str(line1[0:5])+".png "+str(line2))

fileVector = open('/home/sharmin/Desktop/vector.txt','a')

fileData = open('/home/sharmin/Desktop/data.txt','r')
for line in fileData:
	print str(line)
	image = np.array(Image.open(cStringIO.StringIO(urllib2.urlopen("https://s3.amazonaws.com/eds-uga-csci8360/data/project3/images/"+line[0:5]+".png").read())))
	grey = np.zeros((32,32))
	for rownum in range(len(image)):
		for colnum in range(len(image[rownum])):
			grey[rownum][colnum] = average(image[rownum][colnum])
			plt.imshow(grey, cmap = matplotlib.cm.Greys_r)

			flat_image = grey.ravel()
			vector = np.matrix(flat_image)
	fileVector.write(str(line[0:5]+' '+str(vector)+'\n'))

fileVector.close()
