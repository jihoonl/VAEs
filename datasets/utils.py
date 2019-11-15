import threading
from torchvision import transforms

import numpy as np


class ThreadedListHandler(threading.Thread):

    def __init__(self, data, func):
        super(ThreadedListHandler, self).__init__()

        self._data = data
        self._func = func
        self._result = None

    def run(self):
        self._result = self._func(self._data)

    @property
    def result(self):
        return self._result


def threaded_elementwise_operation(data, func, num_thread=8):
    each_size = int(np.ceil(len(data) / num_thread))

    start = 0
    end = each_size

    threads = []
    each_result = []
    for i in range(num_thread):
        each_result.append([])
        t = ThreadedListHandler(data[start:end], func)
        t.start()
        threads.append(t)
        start = end
        end = end + each_size if end + each_size < len(data) else len(data)

    result = []
    for i, t in enumerate(threads):
        t.join()
        result.extend(t.result)
    return result


IMAGE_SIZE = 64


def transform_image(f):
    image_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    return image_transform(f)
