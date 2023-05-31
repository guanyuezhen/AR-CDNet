import numpy
import numpy as np
import torch
import random
import cv2


class Scale(object):
    """
    Resize the given image to a fixed scale
    """

    def __init__(self, wi, he):
        '''
        :param wi: width after resizing
        :param he: height after reszing
        '''
        self.w = wi
        self.h = he

    # modified from torchvision to add support for max size

    def __call__(self, img, label):
        '''
        :param img: RGB image
        :param label: semantic label image
        :return: resized images
        '''
        # bilinear interpolation for RGB image
        img = cv2.resize(img, (self.w, self.h))
        # nearest neighbour interpolation for label image
        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        return [img, label]


class RandomCropResize(object):
    """
    Randomly crop and resize the given image with a probability of 0.5
    """

    def __init__(self, crop_border=60):
        '''
        :param crop_area: area to be cropped (this is the max value and we select between 0 and crop area)
        '''
        self.cw = crop_border
        self.ch = crop_border

    def __call__(self, img, label):
        if random.random() < 0.5:
            h, w = img.shape[:2]
            x1 = random.randint(0, self.ch)
            y1 = random.randint(0, self.cw)

            img_crop = img[y1:h - y1, x1:w - x1]
            label_crop = label[y1:h - y1, x1:w - x1]

            img_crop = cv2.resize(img_crop, (w, h))
            label_crop = cv2.resize(label_crop, (w, h), interpolation=cv2.INTER_NEAREST)

            return img_crop, label_crop
        else:
            return [img, label]


class RandomFlip(object):
    """
    Randomly flip the given Image with a probability of 0.5
    """

    def __call__(self, image, label):
        if random.random() < 0.5:
                image = cv2.flip(image, 0)  # horizontal flip
                label = cv2.flip(label, 0)  # horizontal flip
        if random.random() < 0.5:
                image = cv2.flip(image, 1)  # veritcal flip
                label = cv2.flip(label, 1)  # veritcal flip
        return [image, label]


class RandomExchange(object):
    """
    Randomly exchange bi-temporal images with a probability of 0.5
    """

    def __call__(self, image, label):
        if random.random() < 0.5:
            pre_img = image[:, :, 0:3]
            post_img = image[:, :, 3:6]
            image = numpy.concatenate((post_img, pre_img), axis=2)
        return [image, label]


class Normalize(object):
    """
    Given mean: (B, G, R) and std: (B, G, R),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        '''
        :param mean: global mean computed from dataset
        :param std: global std computed from dataset
        '''
        self.mean = mean
        self.std = std
        self.depth_mean = [0.5]
        self.depth_std = [0.5]

    def __call__(self, image, label):
        image = image.astype(np.float32)
        image = image / 255
        label = np.ceil(label / 255)
        for i in range(6):
            image[:, :, i] -= self.mean[i]
        for i in range(6):
            image[:, :, i] /= self.std[i]

        return [image, label]


class ToTensor(object):
    '''
    This class converts the data to tensor so that it can be processed by PyTorch
    '''

    def __init__(self, scale=1):
        '''
        :param scale: set this parameter according to the output scale
        '''
        self.scale = scale

    def __call__(self, image, label):
        if self.scale != 1:
            h, w = label.shape[:2]
            image = cv2.resize(image, (int(w), int(h)))
            label = cv2.resize(label, (int(w / self.scale), int(h / self.scale)), \
                               interpolation=cv2.INTER_NEAREST)
        image = image[:, :, ::-1].copy()  # .copy() is to solve "torch does not support negative index"
        image = image.transpose((2, 0, 1))
        image_tensor = torch.from_numpy(image)
        # TODO: here, we add unsqueeze to satisfy the condition that
        # adjust_size in DataSet.py should input 4D tensor
        label_tensor = torch.LongTensor(np.array(label, dtype=np.int)).unsqueeze(dim=0)

        return [image_tensor, label_tensor]


class Compose(object):
    """
    Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
