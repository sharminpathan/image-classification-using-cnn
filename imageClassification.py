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

spark = SparkSession\
		.builder\
		.appName("ImageClassification")\
		.getOrCreate()

np.set_printoptions(threshold=np.inf)
def average(pixel):
	return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

data = urllib2.urlopen("https://s3.amazonaws.com/eds-uga-csci8360/data/project3/metadata/X_train.txt")
file = open('/home/sharmin/Desktop/data.txt','a')

for line in data:
	image = np.array(Image.open(cStringIO.StringIO(urllib2.urlopen("https://s3.amazonaws.com/eds-uga-csci8360/data/project3/images/"+line[0:5]+".png").read())))

	grey = np.zeros((32,32))
	for rownum in range(len(image)):
		for colnum in range(len(image[rownum])):
			grey[rownum][colnum] = average(image[rownum][colnum])
			plt.imshow(grey, cmap = matplotlib.cm.Greys_r)

			flat_image = grey.ravel()
			vector = np.matrix(flat_image)
	file.write(str(line[0:5]+' '+str(vector)+'\n'))

file.close()
