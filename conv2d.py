import numpy as np
from layer import Layer

class Conv2D(Layer):
    """
    Conv2D layer searches through the image applying filters in order order to find patterns
    @param image: input image
    @param filters: number of filters
    @param kernel_size: size of the kernel
    @param strides: strides of the convolution
    @param padding: padding of the convolution
    @param activation: activation function  
    """
    def __init__(self, image, filters, kernel, strides=1, padding=0, activation=None):
        super().__init__()
        self.image = image
        self.filters = filters
        self.kernel = kernel
        self.strides = strides
        self.padding = padding
        self.activation = activation

    def forward(self):
        image_H, image_W = self.image.shape
        kernel_H, kernel_W = self.kernel.shape

        if self.padding > 0:
            self.image = np.pad(self.image, ((self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        output_H = (image_H - kernel_H) // self.strides + 1
        output_W = (image_W - kernel_W) // self.strides + 1

        output = np.zeros((output_H, output_W))

        for i in range(0, output_H):
            for j in range(0, output_W):
                region = self.image[i*self.strides:i*self.strides+kernel_H, j*self.strides:j*self.strides+kernel_W]
                output[i, j] = np.sum(region * self.kernel)
        
        if self.activation:
            output = self.activation(output)

        return output

