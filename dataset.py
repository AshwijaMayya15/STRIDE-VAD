import torch.utils.data as data
import numpy as np
import torch
import random
import os
torch.set_float32_matmul_precision('medium')

class SpatioTemporalDataset(data.Dataset):
    """Dataset that loads from separate X3D_Videos and FLOW_Videos folders with class subdirectories"""
    
    def __init__(self, args, test_mode=False):
        if test_mode:
            self.x3d_list_file = args.test_rgb_list
            self.flow_list_file = args.test_flow_list
        else:
            self.x3d_list_file = args.rgb_list
            self.flow_list_file = args.flow_list
            
        self.test_mode = test_mode
        self.use_flow = args.use_flow
        
        # Load file lists
        self.x3d_list = list(open(self.x3d_list_file))
        self.flow_list = list(open(self.flow_list_file))
        
        # Verify both lists have same length
        assert len(self.x3d_list) == len(self.flow_list), \
            f"X3D list ({len(self.x3d_list)}) and Flow list ({len(self.flow_list)}) must have same length"
        
        # Count actual normal videos (those with "Normal" in path)
        self.n_len = sum(1 for path in self.x3d_list if "Normal" in path)
        self.a_len = len(self.x3d_list) - self.n_len
        
        print(f"SpatioTemporalDataset initialized:")
        print(f"  use_flow: {self.use_flow}")
        print(f"  x3d_list_file: {self.x3d_list_file}")
        print(f"  flow_list_file: {self.flow_list_file}")
        print(f"  total files: {len(self.x3d_list)}")
        print(f"  normal files: {self.n_len}")
        print(f"  anomaly files: {self.a_len}")
    
    def load_features(self, x3d_path, flow_path):
        """
        Load features from separate X3D and Flow folders with class subdirectories.
        
        X3D path format: X3D_Videos/Abuse/Abuse001_x264.npy
        Flow path format: FLOW_Videos/Abuse/Abuse001_x264.npy
        
        X3D shape: (192, 16, 10, 10)
        Flow shape: (6, 16, 10, 10)
        Fused shape: (198, 16, 10, 10)
        """
        
        # Load X3D features: (192, 16, 10, 10) = (C, T, H, W)
        x3d = np.load(x3d_path, allow_pickle=True)
        x3d = np.array(x3d, dtype=np.float32)
        
        if self.use_flow:
            # Load Flow features: (6, 16, 10, 10) = (C, T, H, W)
            flow = np.load(flow_path, allow_pickle=True)
            flow = np.array(flow, dtype=np.float32)
            
            # Concatenate along channel dimension: (192+6, 16, 10, 10) = (198, T, H, W)
            features = np.concatenate([x3d, flow], axis=0)
        else:
            features = x3d  # (192, 16, 10, 10)
        
        # Return as (C, T, H, W) - DataLoader will add batch dimension to make (B, C, T, H, W)
        return torch.tensor(features, dtype=torch.float32)
    
    def __getitem__(self, index):
        if not self.test_mode:
            if index == 0:
                self.n_ind = list(range(self.a_len, len(self.x3d_list)))
                self.a_ind = list(range(self.a_len))
                random.shuffle(self.n_ind)
                random.shuffle(self.a_ind)
            
            nindex = self.n_ind.pop()
            aindex = self.a_ind.pop()
            
            # Get paths from both lists
            nx3d_path = self.x3d_list[nindex].strip('\n').strip()
            nflow_path = self.flow_list[nindex].strip('\n').strip()
            
            ax3d_path = self.x3d_list[aindex].strip('\n').strip()
            aflow_path = self.flow_list[aindex].strip('\n').strip()
            
            nfeatures = self.load_features(nx3d_path, nflow_path)  # (C, T, H, W)
            afeatures = self.load_features(ax3d_path, aflow_path)  # (C, T, H, W)
            
            # Label based on "Normal" in path
            nlabel = 0.0 if "Normal" in nx3d_path else 1.0
            alabel = 0.0 if "Normal" in ax3d_path else 1.0
            
            return nfeatures, torch.tensor(nlabel), afeatures, torch.tensor(alabel)
        else:
            # Test mode
            x3d_path = self.x3d_list[index].strip('\n').strip()
            flow_path = self.flow_list[index].strip('\n').strip()
            
            features = self.load_features(x3d_path, flow_path)  # (C, T, H, W)
            
            # Label based on "Normal" in path
            label = 0.0 if "Normal" in x3d_path else 1.0
            return features, torch.tensor(label)
    
    def __len__(self):
        return len(self.x3d_list) if self.test_mode else min(self.a_len, self.n_len)