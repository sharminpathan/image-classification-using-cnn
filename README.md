Team Name: minions

Team Members: Chirag Jain, Sharmin Pathan

Image Classification

Approach: Develop a CNN using Deep Learning to classify CIFAR-10 imageset

Technologies Used:

    Python 2.7
    Caffe
    Apache Spark 2.0
    CUDA
    CuDNN
    
Preprocessing:
- The images were accessed from an LMDB. LMDB is a database when using large datasets. It provides easy access.
- LMDB requires text files that contain the image paths and labels for the training and validation sets.
- Xtrain.txt and Xval.txt have the corressponding image lists extracted from S3 for the training and validation images. 70% of the image dataset was used for training and the rest 30% was used as the validation set.
- LMDBs were created using the following commands:

    <path to caffe>/build/tools/convert_imageset \
    --shuffle \
    <path to the images> \
    Xtrain.txt \
    trainLmdb
    
    <path to caffe>/build/tools/convert_imageset \
    --shuffle \
    <path to the images> \
    Xval.txt \
    valLmdb
    
- The above commands create training and validation LMDBs named trainLmdb and valLmdb respectively.

Flow:
- The above commands create training and validation LMDBs named trainLmdb and valLmdb respectively.

- The project is built around Caffe. Link to install Caffe: http://caffe.berkeleyvision.org/installation.html
- 
