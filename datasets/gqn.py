import collections
import gzip
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import threaded_elementwise_operation, transform_image

Scene = collections.namedtuple('Scene', ['frames', 'cameras'])

DATASETS = [
    'jaco', 'mazes', 'rooms_ring_camera',
    'rooms_free_camera_no_object_rotations',
    'rooms_free_camera_with_object_rotations', 'shepard_metzler_5_parts',
    'shepard_metzler_7_parts'
]
"""
DATASETS = dict(
    jaco=DatasetInfo(basepath='jaco',
                     train_size=3600,
                     test_size=400,
                     frame_size=64,
                     sequence_size=11),
    mazes=DatasetInfo(basepath='mazes',
                      train_size=1080,
                      test_size=120,
                      frame_size=84,
                      sequence_size=300),
    rooms_free_camera_with_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_with_object_rotations',
        train_size=2034,
        test_size=226,
        frame_size=128,
        sequence_size=10),
    rooms_ring_camera=DatasetInfo(basepath='rooms_ring_camera',
                                  train_size=2160,
                                  test_size=240,
                                  frame_size=64,
                                  sequence_size=10),
    rooms_free_camera_no_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_no_object_rotations',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),
    shepard_metzler_5_parts=DatasetInfo(basepath='shepard_metzler_5_parts',
                                        train_size=900,
                                        test_size=100,
                                        frame_size=64,
                                        sequence_size=15),
    shepard_metzler_7_parts=DatasetInfo(basepath='shepard_metzler_7_parts',
                                        train_size=900,
                                        test_size=100,
                                        frame_size=64,
                                        sequence_size=15))
"""


class GQNDataset(Dataset):

    def __init__(self,
                 root_dir,
                 dataset,
                 mode='train',
                 length=None,
                 index=None,
                 use_cache=False):
        self.root_dir = Path(root_dir) / dataset / mode
        if index:
            self.root_dir = self.root_dir / index
        self.dataset = dataset
        self.mode = mode
        self._data = self._get_files(self.root_dir, True if index else False,
                                     length)

    def _get_files(self, root, index=False, use_cache=False, length=None):
        if index:
            return list(root.glob('*.pt.gz'))
        if length:
            cache_file = root / 'cache_{}.pkl'.format(length)
        else:
            cache_file = root / 'cache.pkl'
        if use_cache and cache_file.exists():
            with open(str(cache_file), 'rb') as f:
                data = pickle.load(f)
                return data

        dirs = [x for x in root.iterdir() if root.is_dir()]

        def globber(f):
            t = []
            for each_f in f:
                t.extend(each_f.glob('*.pt.gz'))
            return t

        data = threaded_elementwise_operation(dirs, globber)
        if length:
            data = data[:length]
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        return data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        data_file = str(self._data[idx])

        with gzip.open(data_file, 'r') as f:
            data = torch.load(f)
            frames = data.frames

        [idx] = random.sample(range(len(frames)), 1)
        frame = torch.from_numpy(frames[idx])

        #  Converting uint8 to float. (0, 255) -> (0.0, 1.0)
        frame = frame.permute(2, 0, 1).float() / 255
        frame = transform_image(frame)
        return frame, 0
