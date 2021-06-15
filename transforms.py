import numpy as np
import random


class Window:

    def __init__(self, window_min, window_max):
        self.window_min = window_min
        self.window_max = window_max

    def __call__(self, image):
        image = np.clip(image, self.window_min, self.window_max)

        return image


class MinMaxNorm:

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, image):
        image = (image - self.low) / (self.high - self.low)
        image = image * 2 - 1

        return image

class Noise:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):

        noise = np.random.normal(self.mean, self.std ** 0.5, image.shape)
        out = image + noise
        out = np.clip(out, image.min(), image.max())

        return out

class Flip:

    def __call__(self, image, axis):
        
        #x = random.randint(0,2)
        image = np.flip(image, axis)

        return image