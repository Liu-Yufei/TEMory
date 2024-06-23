import torch
import torch.utils.data as data
import os
import numpy as np
import utils 
from utils import process_feat
class polyp_Dataset(data.Dataset):
    def __init__(self, root_dir, mode,  num_segments, seed=-1, is_normal=True,  sampling='random', transform = None):
        if seed >= 0:
            utils.set_seed(seed)
        self.mode = mode
        self.is_normal = is_normal
        self.num_segments = num_segments
        self.sampling = sampling
        self.dataset = '../File/colon_video_features'
        self.normal_root_path_train = root_dir+"colon_i3d_feature_train_normal"
        self.abnormal_root_path_train = root_dir+"colon_i3d_feature_train_abnormal"

        self.normal_root_path_test = root_dir+"colon_i3d_feature_test_normal"
        self.abnormal_root_path_test = root_dir+"colon_i3d_feature_test_abnormal"
      
        self.transform = transform
        self.list, self.len= self._parse_list()
        self.num_frame = 0
        self.labels = None
    def _parse_list(self):

        normal_file_list = sorted(os.listdir(self.normal_root_path_train))
        abnormal_file_list = sorted(os.listdir(self.abnormal_root_path_train))

        normal_file_list_test = sorted(os.listdir(self.normal_root_path_test))
        abnormal_file_list_test = sorted(os.listdir(self.abnormal_root_path_test))
        if self.mode == 'Train': # train
          if self.is_normal:
              l = [self.normal_root_path_train + '/' + s  for s in normal_file_list]
          else:
              l = [self.abnormal_root_path_train + '/' + s  for s in abnormal_file_list]
        else:
          l = [self.normal_root_path_test + '/' + s  for s in normal_file_list_test] + [self.abnormal_root_path_test + '/' + s  for s in abnormal_file_list_test]
        lenl = len(l)
        return l,lenl
    def __len__(self):
        return self.len
    def __getitem__(self, index):

        label = self.get_label()  # get video level label 0/1
        file_name = self.list[index].strip('\n')
        features = np.load(file_name, allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        if self.transform is not None:
            features = self.transform(features)
        
        if self.mode == 'Test':
          return features, file_name

        features = features.transpose(1, 0, 2)  # [10, B, T, F]

        features = process_feat(features.squeeze(0), self.num_segments)  # divide a video into 32 segments
        features = np.array(features, dtype=np.float32)
        features = np.expand_dims(features, 1)
        return features, label
    def get_label(self):
        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)
        return label
    