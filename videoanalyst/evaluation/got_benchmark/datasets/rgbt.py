from __future__ import absolute_import, print_function, unicode_literals

import os
import glob
import numpy as np
import pandas as pd
import io
import six
from itertools import chain
import pickle

from ..utils.ioutils import download, extract


class RGBT(object):
    r"""`RGBT234 or GTOT Datasets.
    
    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.
    """

    def __init__(self, root_dir):
        super(RGBT, self).__init__()
        self.root_dir = root_dir

        self.valid_seqs = pickle.load(open(os.path.join(self.root_dir+'_CURATION', "meta_data.pkl"), 'rb'))
        self.seq_dirs = [os.path.join(self.root_dir, x[0]) for x in self.valid_seqs
                         if x[0].split('/')[-1] == 'v' or x[0].split('/')[-1] == 'visible']  # del infrared files
        self.seq_dirs = sorted(self.seq_dirs)
        self.anno_files = [x + '.txt' for x in self.seq_dirs]
        self.seq_names = [d[0].split('/')[0] for d in self.valid_seqs]
        self.seq_names = list(set(self.seq_names))
        self.seq_names = sorted(self.seq_names)

    def __getitem__(self, index):
        r"""        
        Args:
            index (integer or string): Index or name of a sequence.
        
        Returns:
            tuple: (img_files, anno), where ``img_files`` is a list of
                file names and ``anno`` is a N x 4 (rectangles) numpy array.
        """
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        img_files = sorted(glob.glob(os.path.join(self.seq_dirs[index], '*.jpg'))
                           +glob.glob(os.path.join(self.seq_dirs[index], '*.png')))

        seq_name = self.seq_names[index]

        # to deal with different delimeters
        anno = np.array(pd.read_csv(self.anno_files[index], sep=',', header=None))
        assert len(img_files) == len(anno)
        assert anno.shape[1] == 4

        return img_files, anno

    def __len__(self):
        return len(self.seq_names)
