import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import threaded_elementwise_operation, transform_image


class CLEVRVAE(Dataset):

    def __init__(self,
                 root_dir='/data/public/rw/datasets/gqn/clevr_context',
                 mode='train',
                 length=None,
                 index=None,
                 use_cache=True):
        """
        Dataset directory structure
        dataset/[0-19]/[0-49]/[0.pkl, 1.pkl, 2.pkl, ..., 1999.pkl]
        """
        self.root_dir = Path(root_dir)
        if index:
            self.root_dir = self.root_dir / index
        self.mode = mode
        self._data = self._get_files(self.root_dir, True if index else False,
                                     mode, use_cache)
        if length:
            self._data = self._data[:length]

    def __len__(self):
        return len(self._data)

    def _get_files(self, root, index, mode, use_cache):
        if index:
            return list(root.glob('*.pkl'))
        cache_file = root / '{}_cache'.format(mode)
        if use_cache and cache_file.exists():
            print('=> CLEVR - Loading data from cache')
            with open(str(cache_file), 'rb') as f:
                data = pickle.load(f)
                print(
                    '=> CLEVR - Loading data from cache. {} {} scenes loaded.'.
                    format(len(data), mode))
                return data

        # Train 0-15, Test 16-19
        if mode == 'train':
            ran = (0, 19)
        else:
            ran = (19, 20)
        dirs = [
            x for x in root.iterdir()
            if root.is_dir() and x.is_dir() and ran[0] <= int(x.name) < ran[1]
        ]
        flatten = []
        for d in dirs:
            flatten.extend(list(d.iterdir()))

        def globber(f):
            t = []
            for each_f in f:
                t.extend(each_f.glob('*.pkl'))
            return t

        logger.info('=> CLEVRGQN - Crawling datafiles...')
        data = threaded_elementwise_operation(flatten, globber)
        logger.info('=> CLEVRGQN - {} {} scenes collected'.format(
            len(data), mode))

        with open(cache_file, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        return data

    def __getitem__(self, idx):
        data_file = str(self._data[idx])
        with open(data_file, 'rb') as f:
            data = pickle.load(f)

        # Extract images from views
        frames = [v['image'] for v in data['views']]
        [idx] = random.sample(range(len(frames)), 1)
        frame = torch.from_numpy(frames[idx])
        frame = frame.permute(2, 0, 1).float() / 255
        frame = transform_image(frame)
        return frame, 0
