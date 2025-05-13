
import numpy as np
import os
import argparse
import os.path
import json
import numpy as np
import pickle as pkl
import csv
from distutils.util import strtobool
import torch
from torch import nn
import torch.backends.cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from collections import OrderedDict
import glob
from tqdm import tqdm
from dominate import document
from dominate.tags import *
from PIL import Image
from matplotlib import pyplot as plt
import trimesh
import cv2
import shutil
from transforms3d.euler import euler2mat
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle


import sys
sys.path.insert(0, os.path.join('/home/user/gs/huggstudy/bite_release', 'src'))

from combined_model.train_main_image_to_3d_wbr_withref import do_validation_epoch
#from combined_model.model_shape_v7_withref_withgraphcnn import ModelImageTo3d_withshape_withproj 

# from combined_model.loss_image_to_3d_withbreedrel import Loss
# from combined_model.loss_image_to_3d_refinement import LossRef
from configs.barc_cfg_defaults import get_cfg_defaults, update_cfg_global_with_yaml, get_cfg_global_updated

from lifting_to_3d.utils.geometry_utils import rot6d_to_rotmat, rotmat_to_rot6d  
from stacked_hourglass.datasets.utils_dataset_selection import get_evaluation_dataset, get_sketchfab_evaluation_dataset, get_crop_evaluation_dataset, get_norm_dict

from test_time_optimization.bite_inference_model_for_ttopt import BITEInferenceModel
from smal_pytorch.smal_model.smal_torch_new import SMAL
from configs.SMAL_configs import SMAL_MODEL_CONFIG
from smal_pytorch.renderer.differentiable_renderer import SilhRenderer
from test_time_optimization.utils.utils_ttopt import reset_loss_values, get_optimed_pose_with_glob

from combined_model.loss_utils.loss_utils import leg_sideway_error, leg_torsion_error, tail_sideway_error, tail_torsion_error, spine_torsion_error, spine_sideway_error
from combined_model.loss_utils.loss_utils_gc import LossGConMesh, calculate_plane_errors_batch
from combined_model.loss_utils.loss_arap import Arap_Loss
from combined_model.loss_utils.loss_laplacian_mesh_comparison import LaplacianCTF     # (coarse to fine animal)
from graph_networks import graphcmr     # .utils_mesh import Mesh
from stacked_hourglass.utils.visualization import save_input_image_with_keypoints, save_input_image

def clip_smal_vals():
    '''
    limit the pose(joint angle) changes for certain joints.
    for example, knee only has 1 degree of freedom.(in skeleton model)
    '''
    limits = np.ones([35, 3, 2])

    limits[..., 0] *= -2
    limits[..., 1] *= 2

    #limits[0] = [[0, 0], [0, 0], [0, 0]]
    
    limits[1] = [[-0.4, 0.4], [-1.0, 0.9], [-0.8, 0.8]]
    limits[2] = [[-0.4, 0.4], [-1.0, 0.9], [-0.8, 0.8]]
    limits[3] = [[-0.4, 0.4], [-0.5, 1.2], [-0.4, 0.4]]
    limits[4] = [[-0.5, 0.5], [-0.4, 1.4], [-0.5, 0.5]]
    limits[5] = [[-0.5, 0.5], [-0.6, 1.4], [-0.8, 0.8]]
    limits[6] = [[-0.5, 0.5], [-0.6, 1.4], [-0.8, 0.8]]

    limits[7] =[[-0.05, 0.05], [-1.3, 0.8], [-0.6, 0.6]]
    limits[8] = [[-0.05, 0.05], [-1.0, 1.1], [-0.6, 0.6]]
    limits[9] = [[-0.4, 0.1], [-0.3, 1.4], [-0.7, 0.4]]
    limits[10] = [[-0.3, 0.1], [-0.4, 1.5], [-0.7, 0.3]]

    limits[11] =[[-0.05, 0.05], [-1.3, 0.8], [-0.6, 0.6]]
    limits[12] = [[-0.05, 0.05], [-1.0, 0.9], [-0.6, 0.6]]
    limits[13] = [[-0.1, 0.4], [-0.3, 1.4], [-0.4, 0.7]]
    limits[14] = [[-0.1, 0.3], [-0.4, 1.5], [-0.3, 0.7]]

    limits[15] = [[-0.8, 0.8], [-1.0, 1.0], [-1.1, 1.1]]
    limits[16] = [[-0.5, 0.5], [-1.0, 0.9], [-0.9, 0.9]]

    limits[17] =[[-0.2, 0.3], [-0.5, 0.8], [-0.5, 0.4]]
    limits[18] = [[-0.2, 0.3], [-0.6, 0.8], [-0.6, 0.5]]
    limits[19] = [[-0.3, 0.2], [-0.8, 0.2], [-0.5, 0.4]]
    limits[20] = [[-0.3, 0.2], [-0.3, 1.1], [-0.5, 0.3]]

    limits[21] =[[-0.3, 0.2], [-0.5, 0.8], [-0.4, 0.5]]
    limits[22] =[[-0.3, 0.2], [-0.6, 0.8], [-0.5, 0.6]]
    limits[23] =[[-0.2, 0.3], [-0.8, 0.2], [-0.4, 0.5]]
    limits[24] =[[-0.2, 0.3], [-0.3, 1.1], [-0.3, 0.5]]


    limits[28] =[[-0.1, 0.1], [-1.0, 1.0], [-0.8, 0.8]]
    limits[29] =[[-0.1, 0.1], [-1.0, 1.0], [-0.8, 0.8]]
    limits[30] =[[-0.1, 0.1], [-1.4, 1.4], [-1.0, 1.0]]
    limits[31] =[[-0.1, 0.1], [-0.7, 1.1], [-0.9, 0.8]]
    
    limits[32] =[[-0.1, 0.1], [-1.1, 0.5], [-0.1, 0.1]]


    
    return limits

"""
def rot_to_rodrigues(rot):
    
    #Converts a rotation matrix to a Rodrigues vector

    #Parameters:
    #rot: Rotation matrix (Bx3x3)

    #Returns:
    #Rodrigues vector (Bx3)
    
    # Ensure the input is a torch tensor
    if not isinstance(rot, torch.Tensor):
        rot = torch.tensor(rot)

    batch_size = rot.shape[0]
    
    # Compute rotation angle and rotation axis
    trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]
    radians = torch.acos((trace - 1) / 2)
    sqrt_denom = 2*torch.sin(radians)

    # Create a rodrigues tensor
    rodrigues = torch.zeros((batch_size, 3),device = rot.device)

    # avoid division by zero
    small_angle = 1e-6

    # Elementwise comparison for small angles
    is_small_angle = radians < small_angle
    sqrt_denom = torch.where(is_small_angle, torch.ones_like(sqrt_denom), sqrt_denom)

    rodrigues[:, 0] = (rot[:, 2, 1] - rot[:, 1, 2]) / sqrt_denom
    rodrigues[:, 1] = (rot[:, 0, 2] - rot[:, 2, 0]) / sqrt_denom
    rodrigues[:, 2] = (rot[:, 1, 0] - rot[:, 0, 1]) / sqrt_denom
    #rodrigues0 = (rot[:, 2, 1] - rot[:, 1, 2]) / sqrt_denom
    #rodrigues1 = (rot[:, 0, 2] - rot[:, 2, 0]) / sqrt_denom
    #rodrigues2 = (rot[:, 1, 0] - rot[:, 0, 1]) / sqrt_denom
    #rodrigues = torch.cat([rodrigues0.unsqueeze(1),rodrigues1.unsqueeze(1),rodrigues2.unsqueeze(1)],dim=1)
    # multiply angle by rotation axis
    rodrigues = rodrigues * radians.unsqueeze(1)
    return rodrigues
"""
def rot_to_rodrigues(rot):
    """
    Converts a rotation matrix to a Rodrigues vector, ensuring compatibility with gradient flow
    by using torch.cat instead of torch.zeros for initialization.

    Parameters:
    rot: Rotation matrix (Bx3x3), assumed to be a torch.Tensor with requires_grad enabled.

    Returns:
    Rodrigues vector (Bx3)
    """
    batch_size = rot.shape[0]
    
    # Compute rotation angle and rotation axis
    trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]
    trace = torch.clamp(trace, -1, 3)  # Clamp trace to avoid numerical issues
    radians = torch.acos((trace - 1) / 2)
    
    # Prevent division by zero for small angles using smooth approximation
    epsilon = 1e-6
    safe_radians = torch.where(radians < epsilon, torch.ones_like(radians) * epsilon, radians)
    sqrt_denom = 2 * torch.sin(safe_radians)
    sqrt_denom = torch.maximum(sqrt_denom, torch.ones_like(sqrt_denom) * epsilon)

    x = (rot[:, 2, 1] - rot[:, 1, 2]) / sqrt_denom
    y = (rot[:, 0, 2] - rot[:, 2, 0]) / sqrt_denom
    z = (rot[:, 1, 0] - rot[:, 0, 1]) / sqrt_denom

    # Multiply angle by rotation axis
    x = x * radians
    y = y * radians
    z = z * radians

    # Use torch.cat to combine the components into the Rodrigues vector
    rodrigues = torch.cat((x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)), dim=1)

    return rodrigues

def rodrigues_to_rot(rodrigues):
    """
    Converts a Rodrigues vector to a rotation matrix

    Parameters:
    rodrigues: Rodrigues vector (Bx3)

    Returns:
    Rotation matrix (Bx3x3)
    """
    # Ensure the input is a torch tensor
    if not isinstance(rodrigues, torch.Tensor):
        rodrigues = torch.tensor(rodrigues)
        
    batch_size = rodrigues.shape[0]

    # Compute rotation axis and rotation angle
    rotation_angle = torch.norm(rodrigues, dim=1, keepdim=True)
    rotation_axis = rodrigues / rotation_angle

    K = torch.zeros((batch_size, 3, 3),device = rodrigues.device)
    K[:, 0, 1] = -rotation_axis[:, 2]
    K[:, 0, 2] = rotation_axis[:, 1]
    K[:, 1, 0] = rotation_axis[:, 2]
    K[:, 1, 2] = -rotation_axis[:, 0]
    K[:, 2, 0] = -rotation_axis[:, 1]
    K[:, 2, 1] = rotation_axis[:, 0]

    # Create identity matrix
    eye = torch.eye(3, device = rodrigues.device).repeat(batch_size, 1, 1)

    # Compute rotation matrix
    rotation_matrix = eye + torch.sin(rotation_angle).unsqueeze(1) * K \
        + (1 - torch.cos(rotation_angle).unsqueeze(1)) * torch.bmm(K, K)

    return rotation_matrix


def rot5deg2radmatrix(rot_animal_z_minus_axis,y_axis,deg):
    vector_a = rot_animal_z_minus_axis
    vector_b = y_axis
    dot_product = np.dot(vector_a, vector_b)

    # Calculate the magnitudes of the vectors
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    # Calculate the cosine of the angle
    cosine_angle = dot_product / (magnitude_a * magnitude_b)

    # Calculate the angle in radians
    angle_radians = np.arccos(cosine_angle)

    # Convert the angle to degrees if needed
    angle_degrees = np.degrees(angle_radians)
    ratio = angle_degrees/90

    root_a = np.deg2rad(deg)/(ratio**2+(1-ratio)**2)**(1/2)
    #print(root_a)
    rad5degaxis = np.array([0.0 , root_a*(1-ratio), root_a*ratio ])
    #print(rad5degaxis)
    return axis_angle_to_matrix(torch.from_numpy(rad5degaxis).float()).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', required=True, type=str, help='the path to the source video')

    opt = parser.parse_args()

    device = 'cuda'
    smal_model_type = '39dogs_norm_newv3' #bite_model.smal_model_type
    logscale_part_list = SMAL_MODEL_CONFIG[smal_model_type]['logscale_part_list']       # ['legs_l', 'legs_f', 'tail_l', 'tail_f', 'ears_y', 'ears_l', 'head_l'] 
    smal = SMAL(smal_model_type=smal_model_type, template_name='neutral', logscale_part_list=logscale_part_list).to(device)    
    silh_renderer = SilhRenderer(image_size=256).to(device)  
    faces_prep = smal.faces.unsqueeze(0).expand((1, -1, -1))



    image_name = opt.image_name
    one_npz_path = f'../GART/data/dog_data_official/{image_name}/pred'
    predz1 = dict(np.load(os.path.join(one_npz_path,'0072.npz')))


    pred_betas = predz1['pred_betas']
    pred_betas_limbs = predz1['pred_betas_limbs']
    pred_pose = predz1['pred_pose']
    pred_trans = predz1['pred_trans']
    pred_focal = predz1['pred_focal']
    pred_betas_limbs_notail = pred_betas_limbs.copy()
    pred_betas_limbs_notail[0][2:4] =-0.300000


    b =pred_pose[0][0] @np.array([0,0,-1])
    y = np.array([0,1,0])
    # 1도로 해야지.
    rot5d = rot5deg2radmatrix(b,y,5)



    geo_list =[]
    deg5_pose = pred_pose.copy()
    for i in range(73,144):
        deg5_pose[0][0] = rot5d@deg5_pose[0][0]
        smal_verts, keyp_3d, _ = smal(beta=torch.from_numpy(pred_betas).to(device), betas_limbs=torch.from_numpy(pred_betas_limbs_notail).to(device), pose=torch.from_numpy(deg5_pose).to(device), trans=torch.from_numpy(pred_trans).float().to(device), keyp_conf='olive', get_skin=True)
        head_dir = keyp_3d[0,16,:] - keyp_3d[0,24,:]
        
        if head_dir[-1]>0:
            #print(head_dir[-1])
            geo_list.append(i)

    ## rotation
    #right_pose_start = min(geo_list)-1
    #left_pose_start = max(geo_list)+1


    at = []
    degn_pose = pred_pose.copy()

    def recursive_dot_product(matrix, idx):
    # 기본 케이스: 횟수가 1이면 원본 행렬을 반환
        if idx == 1:
            return matrix
        # 재귀적으로 이전 결과에 행렬을 dot product 수행
        return matrix@recursive_dot_product(matrix, idx - 1)

    #degn_trans = pred_trans.copy()
    limits = clip_smal_vals()# (105,2)
    limits = torch.from_numpy(limits).float().to(device)


    np.save(f'/home/user/gs/huggstudy/GART/data/dog_data_official/{image_name}/pred/move2.npy',np.array([min(geo_list),max(geo_list)]))
    for i in range(min(geo_list),max(geo_list)+1):
        print(min(geo_list),max(geo_list)+1, i)
        #print(rot5d@rot5d == rot5d**2)
        degn_pose[0][0] =recursive_dot_product(rot5d,(i-72))@pred_pose[0][0]
        smal_verts, keyp_3d, _ = smal(beta=torch.from_numpy(pred_betas).to(device), betas_limbs=torch.from_numpy(pred_betas_limbs_notail).to(device), pose=torch.from_numpy(degn_pose).to(device), trans=torch.from_numpy(pred_trans).float().to(device), keyp_conf='olive', get_skin=True)
        
        #print(degn_trans)
        
        #loss = torch.abs(pred_silh_images[0, 0, :, :] - target_hg_silh)
        degn_trans = pred_trans.copy()
        # divide 2 why -> 중점.
        degn_trans[0][0] = pred_trans[0][0] -float(torch.mean(keyp_3d[0,:24,0]).cpu().numpy())/1.5
        degn_trans[0][1] = pred_trans[0][1] -float(torch.mean(keyp_3d[0,:24,1]).cpu().numpy())/1.5
        degn_trans[0][2] = pred_trans[0][2] -(float(torch.mean(keyp_3d[0,:24,2]).cpu().numpy())-pred_trans[0][2])
        #degn_trans[0][2] = float(torch.mean(keyp_3d[0,:24,2]).cpu().numpy())
        #init_z = degn_trans[0][2].copy()
        #print(init_z)
        #degn_trans2 = torch.tensor(degn_trans, dtype=torch.float,device=device, requires_grad = True)
        smal_verts, keyp_3d, _ = smal(beta=torch.from_numpy(pred_betas).to(device), betas_limbs=torch.from_numpy(pred_betas_limbs_notail).to(device), pose=torch.from_numpy(degn_pose).to(device), trans=torch.from_numpy(degn_trans).float().to(device), keyp_conf='olive', get_skin=True)
        #0pose 추가
        #성공 했긴한데. 뭔가.. 애매하네 ㅋㅋㅋ 
        # 오히려 35x3이 더 잘 안되는듯한 느낌은 뭐지? ㅋㅋ 
        #grad_mask = torch.zeros([1,35,3,3]).float().to(device)
        #
       # grad_mask[0][0] = 1
       # grad_mask[0][3] =1
        #grad_mask[0][8] = 1
        #grad_mask[0][7] = 1
        #grad_mask[0][17] = 1
        #grad_mask[0][18] = 1
        #grad_mask[0][12] = 1
        #grad_mask[0][11] = 1
        #grad_mask[0][21] = 1
        #grad_mask[0][22] = 1
        #grad_mask[0] = 1
        #grad_mask[0] = 1
        #grad_mask[0] = 1
        #grad_mask[0] = 1


        # degn_0pose = torch.tensor(degn_pose, dtype=torch.float,device=device, requires_grad = True)
        # #print(f'degn_pose.shape={degn_pose.shape}')
        # #degn_0pose= rot_to_rodrigues(degn_pose2[0])
        # #degn_0pose = torch.tensor(degn_0pose, dtype=torch.float,device=device, requires_grad = True)
        # #print(f'degn_pose.shape={degn_pose2.shape},degn_0pose.shape={degn_0pose.shape} ')
        # #print(degn_trans.shape)
        # #print(degn_trans[0][0][0:2], degn_trans[0][0][0:2].shape)
        # #optimizer = torch.optim.SGD([degn_trans,degn_0pose], lr=5*1e-3,momentum=0.9)
        # optimizer = torch.optim.SGD([degn_trans2,degn_0pose], lr=5*1e-3,momentum=0.9) #너무 빠름.
        # #optimizer = torch.optim.SGD([degn_trans,degn_0pose], lr=5*1e-4,momentum=0.9) #너무 느림
        # #optimizer = torch.optim.Adam([degn_trans,degn_0pose], lr=0.001,betas=(0.9, 0.999)) 별로..
        # target_hg_silh = torch.tensor(cv2.resize((np.load(f'../GART/data/dog_data_official/{image_name}/images/{i:04d}.npy')),(256,256))).to(device)
        # zeross = torch.zeros([35,3], device=device, dtype =torch.float)
        # #original -> smal vertex를 얻기위해 smal을 한번 더 사용 . but 필요 없으니.
        # loop = tqdm(range(100))
        # for j in loop:
        #     optimizer.zero_grad()
        #     #pose
        #     smal_verts, keyp_3d, _ = smal(beta=torch.from_numpy(pred_betas).to(device), betas_limbs=torch.from_numpy(pred_betas_limbs_notail).to(device), pose=degn_0pose, trans=degn_trans2, keyp_conf='olive', get_skin=True)
        #     #theta
        #     #smal_verts, keyp_3d, _ = smal(beta=torch.from_numpy(pred_betas).to(device), betas_limbs=torch.from_numpy(pred_betas_limbs_notail).to(device), theta=degn_0pose, trans=degn_trans, keyp_conf='olive', get_skin=True)
        #     pred_silh_images, pred_keyp_raw = silh_renderer(vertices=smal_verts, points=keyp_3d, faces=faces_prep, focal_lengths=torch.from_numpy(pred_focal).to(device))

            
        #     loss = torch.mean(torch.abs(pred_silh_images[0, 0, :, :] - target_hg_silh)) #sum하니까 줄어들지 않는데 mean 하니까 줄어든다 왜그러지? 허허...
        #     print(f'{j}_loss = {loss}, degn_trans2={degn_trans2}')
        #     #print(rot_to_rodrigues(degn_0pose[0]))
        #     loss += 0.01*torch.exp(torch.mean(torch.max(rot_to_rodrigues(degn_0pose[0])-limits[...,1],zeross) + torch.max(limits[...,0]-rot_to_rodrigues(degn_0pose[0]),zeross))) #너무 크다.
        #     print(f'{j}_loss+reg = {loss}, degn_trans2={degn_trans2}')
        #     print(f'degn_0pose.shape={degn_0pose.shape}')
        #    #loss.backward()
        #     #차이 확인하기.


        #     #for opti visualization.
        #     im =Image.open(f'../GART/data/dog_data_official/{image_name}/images/{i:04d}.png')
        #     im = im.resize((256,256))
        #     tt = np.array(im).astype(np.float32)
        #     tt = tt/255.0
        #     visualizations = silh_renderer.get_visualization_nograd(smal_verts, faces_prep, torch.from_numpy(pred_focal).to(device), color=0)
        #     pred_tex = visualizations[0, :, :, :].permute((1, 2, 0)).cpu().detach().numpy() / 256
        #     im_masked = cv2.addWeighted(tt,0.2,pred_tex,0.8,0)
        #     pred_tex_max = np.max(pred_tex, axis=2)
        #     im_masked[pred_tex_max<0.01, :] = tt[pred_tex_max<0.01, :]
        #     out_path = f'/home/user/gs/huggstudy/GART/data/dog_data_official/{image_name}/pred/test.png'
        #     plt.imsave(out_path, im_masked,format='png')

        #     #기존 그래프를 계속유지..
        #     loss.backward(retain_graph=True)
        #     #print(f'degn_0pose={degn_0pose[0][0]}')
        #     #loss.backward()
        #     degn_0pose.grad = degn_0pose.grad  * grad_mask
        #     optimizer.step()
    
        # #pred_keyp = pred_keyp_raw[:, :24, :]

        # #img_silh = Image.fromarray(np.uint8(255*pred_silh_images[0, 0, :, :].detach().cpu().numpy())).convert('RGB')
        # #at.append(img_silh)'
        # #degn_trans[0][2]=torch.tensor(init_z).to(device=device, dtype=degn_trans.dtype)
        # im =Image.open(f'../GART/data/dog_data_official/{image_name}/images/{i:04d}.png')
        # im = im.resize((256,256))
        # tt = np.array(im).astype(np.float32)
        # tt = tt/255.0

        visualizations = silh_renderer.get_visualization_nograd(smal_verts, faces_prep, torch.from_numpy(pred_focal).to(device), color=0)
        #masks, _ = silh_renderer(vertices=smal_verts, points=keyp_3d, faces=faces_prep, focal_lengths=torch.from_numpy(pred_focal).to(device))
        #print(visualizations.shape) torch.Size([1, 3, 256, 256]) 뭐지 ..pred_tex는 왜 4채널이 된거지?
        pred_tex = visualizations[0, :, :, :].permute((1, 2, 0)).cpu().detach().numpy() / 256
        #print(pred_tex.shape) 
        #(256, 256, 3) 인데 뭐지?? 
        # out_path = root_out_path_details +  name + '_tex_pred_e' + format(i, '03d') + '.png'
        # plt.imsave(out_path, pred_tex) 

        #im_masked = cv2.addWeighted(tt,0.2,pred_tex,0.8,0)
        #pred_tex_max = np.max(pred_tex, axis=2)
        #im_masked[pred_tex_max<0.01, :] = tt[pred_tex_max<0.01, :]
        #out_path = f'/home/user/gs/huggstudy/GART/data/dog_data_official/{image_name}/pred/{i:04d}.png'
        #plt.imsave(out_path, im_masked,format='png')

        #tt2 = np.ones((256,256,3)).astype(np.float32)
        #im_masked2 = cv2.addWeighted(tt2,0.2,pred_tex,0.8,0) # pred_tex가 4채널이라;; 이렇게 된거였구나.. 허허..
        #pred_tex_max2 = np.max(pred_tex, axis=2)
        #im_masked2[pred_tex_max2<0.01, :] = tt2[pred_tex_max2<0.01, :]
        out_path2 = f'/home/user/gs/huggstudy/GART/data/dog_data_official/{image_name}/pred/{i:04d}_only.png'
        plt.imsave(out_path2, pred_tex,format='png')
        mask_d = np.zeros((256,256))
        vi = visualizations.cpu().detach().numpy() 
        #print(vi)
        #np.save('/home/user/gs/huggstudy/test.npy',vi)
        mask_d[vi[0,1,:,:]>1.0000000001] = 1
        np.save(f'/home/user/gs/huggstudy/GART/data/dog_data_official/{image_name}/pred/{i:04d}_mask.npy',mask_d)
        #print(f'final_degn_trans2={degn_trans2.detach().cpu().numpy()}')  #잘 들어가는데 뭐가 문제일까.
        #if i ==133:
        #    break
        np.savez(f'/home/user/gs/huggstudy/GART/data/dog_data_official/{image_name}/pred/{i:04d}.npz',pred_betas=pred_betas, pred_betas_limbs=pred_betas_limbs_notail, pred_pose=degn_pose, pred_trans=degn_trans, pred_focal=pred_focal)



if __name__ == "__main__":
    main()
