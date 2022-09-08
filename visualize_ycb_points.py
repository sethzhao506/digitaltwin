import _init_paths
import argparse
import os
import copy
import random
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import numpy.ma as ma
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset
from lib.network import PoseNet, PoseRefineNet
import cv2
from lib.randla_utils import randla_processing
from cfg.config import YCBConfig as Config, write_config
from DenseFusionModule import DenseFusionModule

from lib.loss import Loss

try:
    from lib.tools import compute_rotation_matrix_from_ortho6d
except:
    from tools import compute_rotation_matrix_from_ortho6d

import open3d as o3d

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default = 'ckpt/df-epoch=25-val_dis=0.01744.ckpt',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=bool, default = False,  help='resume PoseRefineNet model')
parser.add_argument('--output', type=str, default='visualization', help='output for point vis')
parser.add_argument('--use_posecnn_rois', action="store_true", default=False, help="use the posecnn roi's")
opt = parser.parse_args()

cfg = Config()
cfg.refine_start = opt.refine_model != ''

if opt.use_posecnn_rois:
    from datasets.ycb.dataset import PoseDatasetPoseCNNResults as PoseDataset
else:
    from datasets.ycb.dataset import PoseDatasetAllObjects as PoseDataset

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])
cam_cx = 312.9869
cam_cy = 241.3109
cam_fx = 1066.778
cam_fy = 1067.487
cam_scale = 10000.0
num_obj = 21
img_width = 480
img_length = 640
num_points = 1000
num_points_mesh = 500
iteration = 2
batch_size = 1
workers = 1

cfg.posecnn_results = "YCB_Video_toolbox/results_PoseCNN_RSS2018"

def get_pointcloud(model_points, t, rot_mat):

    model_points = model_points.cpu().detach().numpy()

    pts = (model_points @ rot_mat.T + t).squeeze()

    return pts

def project_points(pts, cam_fx, cam_fy, cam_cx, cam_cy):
    proj_mat = np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
    projected_pts = pts @ proj_mat.T
    projected_pts /= np.expand_dims(projected_pts[:,2], -1)
    projected_pts = projected_pts[:,:2]
    return projected_pts

def main():
    if not os.path.isdir(opt.output):
        os.mkdir(opt.output)

    df_module = DenseFusionModule.load_from_checkpoint(cfg = cfg, checkpoint_path = opt.checkpoint_path)
    estimator = df_module.estimator

    # estimator = nn.DataParallel(estimator)
    estimator.cuda()
    estimator.eval()

    
    if opt.refine_model:
        refiner = df_module.refiner
        refiner.cuda()
        refiner.eval()

    if opt.use_posecnn_rois:
        test_dataset = PoseDataset('test', cfg = cfg)
    else:
        test_dataset = PoseDataset('test', cfg = cfg)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    colors = [(96, 60, 20), (156, 39, 6), (212, 91, 18), (243, 188, 46), (95, 84, 38)]

    with torch.no_grad():

        for now, data_objs in enumerate(test_dataloader):

            print("frame: {0}".format(now))

            color_img_file = '{0}/{1}-color.png'.format(cfg.root, test_dataset.list[now])
            color_img = cv2.imread(color_img_file)

            for obj_idx, end_points in enumerate(data_objs):
                torch.cuda.empty_cache()

                intr = end_points["intr"]
                del end_points["intr"]

                end_points_cuda = {}
                for k, v in end_points.items():
                    end_points_cuda[k] = Variable(v).cuda()

                end_points = end_points_cuda

                if cfg.pcld_encoder == "randlanet":
                    end_points = randla_processing(end_points, cfg)

                cam_fx, cam_fy, cam_cx, cam_cy = [x.item() for x in intr]
                                                                        
                end_points = estimator(end_points)

                pred_r = end_points["pred_r"]
                pred_t = end_points["pred_t"]
                points = end_points["cloud"] + end_points["cloud_mean"]
                model_points = end_points["model_points"]

                bs, num_p, _ = pred_t.shape

                if cfg.use_normals:
                    normals = end_points["normals"]

                if cfg.use_confidence:
                    pred_c = end_points["pred_c"]
                    pred_c = pred_c.view(bs, num_p)
                    how_max, which_max = torch.max(pred_c, 1)
                    pred_t = pred_t.view(bs * num_p, 1, 3)

                    my_r = pred_r[0][which_max[0]].view(-1).unsqueeze(0).unsqueeze(0)

                    my_rot_mat = compute_rotation_matrix_from_ortho6d(my_r)[0].cpu().data.numpy()

                    points = points.contiguous().view(bs*num_p, 1, 3)

                    my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
                else:

                    my_r = torch.mean(pred_r, dim=1, keepdim=True)
                    pred_t = points + pred_t
                    my_t = torch.mean(pred_t, dim=1).view(-1).cpu().data.numpy()
                    my_rot_mat = compute_rotation_matrix_from_ortho6d(my_r)[0].cpu().data.numpy()

                if opt.refine_model:
                    for ite in range(0, cfg.iteration):
                        T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_p, 1).contiguous().view(1, num_p, 3)
                        rot_mat = my_rot_mat

                        #print('iter!', ite, my_rot_mat)

                        my_mat = np.zeros((4,4))
                        my_mat[0:3,0:3] = rot_mat
                        my_mat[3, 3] = 1

                        #print(my_mat, my_mat.shape)

                        R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                        my_mat[0:3, 3] = my_t

                        points = points.view(bs, num_p, 3)
                        
                        new_points = torch.bmm((points - T), R).contiguous()
                        end_points["new_points"] = new_points.detach()

                        if cfg.use_normals:
                            normals = normals.view(bs, num_p, 3)
                            new_normals = torch.bmm(normals, R).contiguous()
                            end_points["new_normals"] = new_normals.detach()

                        end_points = refiner(end_points, ite)


                        pred_r = end_points["refiner_pred_r_" + str(ite)]
                        pred_t = end_points["refiner_pred_t_" + str(ite)]

                        pred_r = pred_r.view(1, 1, -1)
                        pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                        my_r_2 = pred_r.view(-1).unsqueeze(0).unsqueeze(0)
                        my_t_2 = pred_t.view(-1).cpu().data.numpy()
                        rot_mat_2 = compute_rotation_matrix_from_ortho6d(my_r_2)[0].cpu().data.numpy()

                        my_mat_2 = np.zeros((4, 4))
                        my_mat_2[0:3,0:3] = rot_mat_2
                        my_mat_2[3,3] = 1
                        my_mat_2[0:3, 3] = my_t_2

                        my_mat_final = np.dot(my_mat, my_mat_2)
                        my_r_final = copy.deepcopy(my_mat_final)
                        my_rot_mat[0:3,0:3] = my_r_final[0:3,0:3]
                        my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

                        #my_pred = np.append(my_r_final, my_t_final)
                        my_t = my_t_final
                # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)

                my_r = copy.deepcopy(my_rot_mat)
                pred = get_pointcloud(model_points, my_t, my_r)

                projected_pred = project_points(pred, cam_fx, cam_fy, cam_cx, cam_cy)

                r, g, b = colors[obj_idx % len(colors)]

                for (x, y) in projected_pred:
                    color_img = cv2.circle(color_img, (int(x), int(y)), radius=1, color=(b,g,r), thickness=-1)

            output_filename = '{0}/{1}.png'.format(opt.output, now)
            cv2.imwrite(output_filename, color_img)

if __name__ == "__main__":
    main()
