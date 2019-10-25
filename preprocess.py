import torch


class Dummy(object):

    def preprocess(self, image):
        return image

    def postprocess(self, image):
        return image


# Image Quantization
class Quantization(object):

    def __init__(self, bits=5, device='cpu'):
        self.bits = 5
        self.device = device

    def preprocess(self, image):
        bits = self.bits

        bins = 2**bits
        image = image * 255.0
        if bits < 8:
            image = torch.floor(image / 2**(8 - bits))
        image = image / bins
        image = image + torch.rand(image.size()).to(self.device) / bins
        image = image - 0.5
        return image * 2.0

    def postprocess(self, image):
        bits = self.bits
        bins = 2**bits
        image = image / 2.0 + 0.5
        image = torch.floor(bins * image)
        image = image * (255.0 / (bins - 1))
        image = torch.clamp(image, min=0, max=255) / 255.0
        return image
