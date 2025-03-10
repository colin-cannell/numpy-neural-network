import numpy as np
from layer import Layer

class MaxPool(Layer):
    """
    MaxPool layer reduces the size of the image by taking the maximum value in each region
    @param image: input image
    @param pool_size: size of the pooling region
    @param strides: strides of the pooling
    """
    def __init__(self, image, pool_size=2, strides=2):
        self.image = image
        self.pool_size = pool_size
        self.strides = strides
    
    def forward(self):
        image_H, image_W = self.image.shape

        output_H = (image_H - self.pool_size) // self.strides + 1
        output_W = (image_W - self.pool_size) // self.strides + 1

        output = np.zeros((output_H, output_W))

        for i in range(0, output_H):
            for j in range(0, output_W):
                region = self.image[i*self.strides:i*self.strides+self.pool_size, j*self.strides:j*self.strides+self.pool_size]
                output[i, j] = np.max(region)

        return output
    