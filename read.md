Team name: minions

Team members: Chirag Jain, Sharmin Pathan

Project 3 - Image Classification

Approach: Develop a CNN using Deep Learning to classify CIFAR-10 imageset

Technologies Used:
- Python 2.7
- Caffe
- Apache Spark 2.0
- CUDA
- CuDNN

Preprocessing:
- The images were accessed from LMDB. LMDB is a database when using large datasets. It provides easy access.
- LMDB requires text files that contain image paths and labels for the training and validation sets.
- Xtrain.txt and Xval.txt contain the image lists extracted from S3 for the training and validation images. 70% of the X_train dataset was used for training and the rest 30% for validation.
- LMDBs were created using the following commands:

  path-to-caffe/build/tools/convert-imageset \
  --shuffle \
  path-to-images \
  Xtrain.txt \
  trainLmdb
  
  path-to-caffe/build/tools/convert-imageset \
  --shuffle \
  path-to-images \
  Xval.txt \
  valLmdb
  
- The above commands create trainLmdb and valLmdb for training and validation respectively.
- The next step is to compute the image mean for the training set which is used for both training and prediction. The image mean is computed using the following command:

  path-to-caffe/build/tools/compute_image_mean trainLmdb mean.binaryproto
  

  
