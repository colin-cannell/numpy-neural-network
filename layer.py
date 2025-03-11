class Layer:
    """
    Base class for all layers
    """
    def __init__(self):
        pass

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        pass