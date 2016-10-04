import cv2
import urllib2
from matplotlib import pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.inf)
data = urllib2.urlopen("https://s3.amazonaws.com/eds-uga-csci8360/data/project3/metadata/X_small_train.txt")

fileEdge = open('/home/sharmin/Desktop/fileEdge.txt','a')
for line in data:
	req = urllib2.urlopen('https://s3.amazonaws.com/eds-uga-csci8360/data/project3/images/00000.png')
	arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
	image = cv2.imdecode(arr,-1)

	edges = cv2.Canny(image,100,200)
	plt.subplot(122),plt.imshow(edges,cmap = 'gray')
	plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
 
	fileEdge.write(str(line[0:5])+str(edges.ravel())+'\n')

fileEdge.close()