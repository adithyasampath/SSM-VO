import torch
import sys
sys.path.append("../")
from rotation_conversions import matrix_to_axis_angle, axis_angle_to_matrix

class PPNetUtils:
    def __init__(self, device):
        self.device = device
        
    def se2SE(self, se_data):
            SE = torch.eye(4).to(self.device)
            SE[:3, :3] = axis_angle_to_matrix(se_data[3:])
            SE[:3, 3] = se_data[:3]
            return SE

    def ses2SEs(self, se_datas):
        N = se_datas.shape[0]

        SEs = torch.zeros(N, 4, 4).to(self.device)
        for i in range(N):
            SEs[i] = self.se2SE(se_datas[i])
        return SEs

    def SE2se(self, SE_data):
        se = torch.zeros(6).to(self.device)
        se[:3] = SE_data[:3, 3]
        se[3:] = matrix_to_axis_angle(SE_data[:3, :3])
        return se

    def SEs2ses(self, SE_datas):
        N = SE_datas.shape[0]

        ses = torch.zeros(N, 6).to(self.device)
        for i in range(N):
            ses[i] = self.SE2se(SE_datas[i])
        return ses

    # pos_seqs: [BS, 19, 6], relative poses
    def translate_poses(self, pos_seqs, mode='middle'):
        b, num, dim = pos_seqs.shape

        I_idx = (num+1) // 2
        if mode != 'middle':
            print('You need implemente other modes by yourself !')
            return None

        pos_seqs_mat = self.ses2SEs(pos_seqs.reshape(-1, 6)).reshape(b, -1, 4, 4)       # [BS, 19, 4, 4]
        translated_pos_mat = torch.zeros(b, num+1, 4, 4).to(self.device)                # [BS, 20, 4, 4]

        translated_pos_mat[:, I_idx] = torch.eye(4)
        for i in range(I_idx-1, -1, -1):
            translated_pos_mat[:, i] = translated_pos_mat[:, i+1] @ pos_seqs_mat[:, i]
        for i in range(I_idx+1, num+1):
            translated_pos_mat[:, i] = translated_pos_mat[:, i-1] @ pos_seqs_mat[:, i-1].inverse()
        
        translated_poses = self.SEs2ses(translated_pos_mat.reshape(-1, 4, 4)).reshape(b, -1, 6)
        
        return translated_poses

    def get_pseudo_poses(self, frame_ids, use_weak_supervision):
        """Obtain the pseudo label
        1. predict the pseudo pose using PPnet
        2. compute the pseudo label
        """
        # Note that the poses saved in Pose Manager are relative poses
        poses = torch.zeros(self.batch_size, 19, 6).to(self.device)
        for i in range(self.batch_size):
            if use_weak_supervision[i]:
                target_frame_id = frame_ids[i]
                poses[i] = poselist[i][target_frame_id-19:target_frame_id, 0]
        
        # Convert relative poses to pose-centralized poses
        translated_poses = self.translate_poses(poses)      # [BS, 20, 6]

        # TODO: use in trainer
        # 1. predict the pseudo pose using PPnet
        # mean, log_variance = self.models['ppnet'](translated_poses)

        # 2. compute the pseudo label
        # relative_mat_label = self.ses2SEs(mean).inverse() @ self.ses2SEs(translated_poses[:, -1])
        
        # variance = torch.exp(log_variance)

        # return relative_mat_label, variance