from __future__ import print_function, absolute_import
import torch
import numpy as np
# p_pids = [15, 15, 15, 15, 15]
# g_pids = [11, 12, 13, 14, 15]
# g_pids = np.array(g_pids)
# p_pids = np.array(p_pids)
# # print(pids)
# distmat = [[ 6, 4, 8, 3, 2],
#            [ 5, 6, 7, 8, 9],
#            [10, 11, 12, 13, 14],
#            [15, 16, 17, 18, 19]]
# # print(distmat)
# indices = np.argsort(distmat, axis=1)
# print(indices)
# # print(pids[indices])
# pre_label = g_pids[indices]
# matches = (g_pids[indices] == p_pids[:, np.newaxis])
# print(matches)

# feat = np.arange(120).reshape(4, 30)
# feat = torch.from_numpy(feat)
# ptr = 0
# batch_num = feat.size(0)
# gall_feat_pool = np.zeros((4, 30))
# print(feat)
# gall_feat_pool[ptr:ptr+batch_num,: ] = feat.detach()
# print(gall_feat_pool)
metrics = {'Rank-1':[], 'mAP': [], 'mINP': [], 'Rank-5':[], 'Rank-10':[], 'Rank-20':[]}
print(type(metrics))
