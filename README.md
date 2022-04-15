# Handwritten Symbol Recognition

Pytorch and CUDA dependencies needed.

Use the train.py file to train the network.
Test the results with the test.py file


# Data Information
There are 25 different symbols each labeled from 0 to 24. The Images are shuffled and not in order, but the labels line up with the index of the images. The labels of course idicate the true class of the image.

Image and label files included are a sample due to the size of upload file limits on Github. The original used 23503 images. Here I have trimmed that to 5000.
Images are in .npy format and dimensions are (x, 150, 150) where x is the amount of images. and 150x150 is the size of Images.
Labels are (x,) where x is the is the amount of labels

# Network Design
A visual representation of the convolutional network architecture:
<img width="551" alt="image" src="https://user-images.githubusercontent.com/89211293/163593353-689308ca-0e41-4df2-8ac4-f2a20a2a6acb.png">

