# image-search-clothing
Algorithm for finding similar products given an image.

This is done with a technique called vectorization. Using a pre-trained (ResNet50) model without the top layer, you can extract an image`s feature vector, and with it you can compare similarities with other images using vector similarity.

## Dataset
Dataset used was a fashion products images, small for ease of training:

https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

## Results
![fig1](/results/fig1.png)
![fig2](/results/fig2.png)
![fig3](/results/fig3.png)
