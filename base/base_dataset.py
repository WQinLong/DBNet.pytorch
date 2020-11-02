# -*- coding: utf-8 -*-
# @Time    : 2019/12/4 13:12
# @Author  : zhoujun
import copy
from torch.utils.data import Dataset
from data_loader.modules import *


class BaseDataSet(Dataset):

    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None,
                 target_transform=None):
        assert img_mode in ['RGB', 'BRG', 'GRAY']
        self.ignore_tags = ignore_tags
        self.data_list = self.load_data(data_path)
        item_keys = ['img_path', 'img_name', 'text_polys', 'texts', 'ignore_tags']
        for item in item_keys:
            assert item in self.data_list[0], 'data_list from load_data must contains {}'.format(item_keys)
        self.img_mode = img_mode
        self.filter_keys = filter_keys
        self.transform = transform
        self.target_transform = target_transform
        self._init_pre_processes(pre_processes)

    def _init_pre_processes(self, pre_processes):
        self.aug = []
        if pre_processes is not None:
            for aug in pre_processes:
                if 'args' not in aug:
                    args = {}
                else:
                    args = aug['args']
                if isinstance(args, dict):
                    cls = eval(aug['type'])(**args)
                else:
                    cls = eval(aug['type'])(args)
                self.aug.append(cls)

    def load_data(self, data_path: str) -> list:
        """
        把数据加载为一个list：
        :params data_path: 存储数据的文件夹或者文件
        return a dict ,包含了，'img_path','img_name','text_polys','texts','ignore_tags'
        """
        raise NotImplementedError

    def apply_pre_processes(self, data):
        for aug in self.aug:
            data = aug(data)
        return data

    def __getitem__(self, index):
        try:
            data = copy.deepcopy(self.data_list[index])
            im = cv2.imread(data['img_path'], 1 if self.img_mode != 'GRAY' else 0)
            # for box in data['text_polys']:
            #     # if box.shape[0] != 4:
            #     if 1:
            #         cv2.imwrite("./tmp_output/"+str(index)+"_img_origin.jpg", im)
            #         break
            if self.img_mode == 'RGB':
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            data['img'] = im
            data['shape'] = [im.shape[0], im.shape[1]]
            data = self.apply_pre_processes(data)

            if self.transform:
                data['img'] = self.transform(data['img'])
            # data['text_polys'] = data['text_polys'].tolist()
            if len(self.filter_keys):
                data_dict = {}
                for k, v in data.items():
                    if k not in self.filter_keys:
                        data_dict[k] = v
                # for box in data['text_polys']:
                #     # if box.shape[0] != 4:
                #     if 1:
                #         import torch
                #         dtype = data_dict['img'].dtype
                #         mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=dtype, device=data_dict['img'].device)
                #         std = torch.as_tensor([0.229, 0.224, 0.225], dtype=dtype, device=data_dict['img'].device)
                #         if mean.ndim == 1:
                #             mean = mean[:, None, None]
                #         if std.ndim == 1:
                #             std = std[:, None, None]
                #         img_np = data_dict['img']
                #         img_np = img_np.mul_(std).add_(mean)
                #         img_np = img_np.data.cpu().numpy()
                #         img_np = img_np.transpose(1, 2, 0)
                #         # img_np = data_dict['img'].transpose(1, 2, 0).mul_(std).add_(mean).data.cup().numpy()
                #
                #         cv2.imwrite("./tmp_output/"+str(index)+"_img.jpg", cv2.cvtColor((img_np*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                #         threshold_map = data_dict['threshold_map']
                #         _threshold_map = (threshold_map * 255).astype(np.uint8)
                #         cv2.imwrite("./tmp_output/"+str(index)+"_threshold_map.jpg", (threshold_map * 255).astype(np.uint8))
                #         threshold_map = data_dict['threshold_mask']
                #         cv2.imwrite("./tmp_output/"+str(index)+"_threshold_mask.jpg", (threshold_map * 255).astype(np.uint8))
                #         threshold_map = data_dict['shrink_map']
                #         cv2.imwrite("./tmp_output/"+str(index)+"_shrink_map.jpg", (threshold_map * 255).astype(np.uint8))
                #         threshold_map = data_dict['shrink_mask']
                #         cv2.imwrite("./tmp_output/"+str(index)+"_shrink_mask.jpg", (threshold_map * 255).astype(np.uint8))
                #         break
                return data_dict
            else:
                return data
        except:
            return self.__getitem__(np.random.randint(self.__len__()))

    def __len__(self):
        return len(self.data_list)
