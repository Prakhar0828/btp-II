import os
import glob
import json
from torch.utils.data import Dataset


class MVCDataset(Dataset):
    def __init__(self, dirpath, transforms, mvc_info='mvc_info.json', attr_labels='attribute_labels.json'):
        # read the whole data
        dirs = os.walk(dirpath)
        self.f1 = json.loads(open(mvc_info).read())
        self.f2 = json.loads(open(attr_labels).read())

    def __getitem__(self, idx):
        a, b = self.f1[idx], self.f2[idx]
        imgpath = a['image_url_4x'].replace('http://www.zappos.com', 'dataset')

        if not os.path.exists(imgpath):
            return '1'

        f1_keys = ['viewId', 'colourId', 'productTypeId', 'styleId',
                   'total_style', 'productId', 'price', 'brandId', 'catNum']
        f2_del_keys = ['filename', 'itemN', ]
        for i in f2_del_keys:
            if i in b:
                del b[i]
        a = {i: [i] for i in f1_keys}
        return a, b, imgpath

    def __len__(self):
        return len(self.f1)
