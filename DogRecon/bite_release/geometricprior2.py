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

    # for alpha
    one_npz_path = f'../GART/data/dog_data_official/{image_name}/pred'
    predz1 = dict(np.load(os.path.join(one_npz_path,'0000.npz')))


    pred_betas = predz1['pred_betas']
    pred_betas_limbs = predz1['pred_betas_limbs']
    pred_pose = predz1['pred_pose']
    pred_trans = predz1['pred_trans']
    pred_focal = predz1['pred_focal']
    pred_betas_limbs_notail = pred_betas_limbs.copy()
    pred_betas_limbs_notail[0][2:4] =-0.300000


    b =pred_pose[0][0] @np.array([0,0,-1])
    y = np.array([0,1,0])
    rot5d = rot5deg2radmatrix(b,y,5)



    smal_verts, keyp_3d, _ = smal(beta=torch.from_numpy(pred_betas).to(device), betas_limbs=torch.from_numpy(pred_betas_limbs_notail).to(device), pose=torch.from_numpy(pred_pose).to(device), trans=torch.from_numpy(pred_trans).float().to(device), keyp_conf='olive', get_skin=True)

    # render silhouette and keypoints
    pred_silh_images, pred_keyp_raw = silh_renderer(vertices=smal_verts, points=keyp_3d, faces=faces_prep, focal_lengths=torch.from_numpy(pred_focal).to(device))
    pred_keyp = pred_keyp_raw[:, :24, :]



    at = []
    degn_pose = pred_pose.copy()
    #degn_trans = pred_trans.copy()
    rotminus5d= np.linalg.inv(rot5d)


    geo_list =[]
    deg5_pose = pred_pose.copy()
    for i in range(73):
        deg5_pose[0][0] = rot5d@deg5_pose[0][0]
        smal_verts, keyp_3d, _ = smal(beta=torch.from_numpy(pred_betas).to(device), betas_limbs=torch.from_numpy(pred_betas_limbs_notail).to(device), pose=torch.from_numpy(deg5_pose).to(device), trans=torch.from_numpy(pred_trans).float().to(device), keyp_conf='olive', get_skin=True)
        head_dir = keyp_3d[0,16,:] - keyp_3d[0,24,:]
        
        if head_dir[-1]>0:
            #print(head_dir[-1])
            geo_list.append(i)

    ## rotation
    right_pose_start = min(geo_list)-1
    left_pose_start = max(geo_list)+1
    pointer = 0
    for i in range(len(geo_list)):
        # right rotation
        if i < len(geo_list)//2 :


            predz_right = dict(np.load(os.path.join(one_npz_path,f'{right_pose_start:04d}.npz')))


            pred_betas = predz_right['pred_betas']
            pred_betas_limbs = predz_right['pred_betas_limbs']
            pred_pose = predz_right['pred_pose']
            pred_trans = predz_right['pred_trans']
            pred_focal = predz_right['pred_focal']
            pred_betas_limbs_notail = pred_betas_limbs.copy()
            pred_betas_limbs_notail[0][2:4] =-0.300000


            b =pred_pose[0][0] @np.array([0,0,-1])
            y = np.array([0,1,0])
            rot5d = rot5deg2radmatrix(b,y,5)
            if i==0:
                degn_pose = pred_pose.copy()
            degn_pose[0][0] =rot5d@degn_pose[0][0]
            smal_verts, keyp_3d, _ = smal(beta=torch.from_numpy(pred_betas).to(device), betas_limbs=torch.from_numpy(pred_betas_limbs_notail).to(device), pose=torch.from_numpy(degn_pose).to(device), trans=torch.from_numpy(pred_trans).float().to(device), keyp_conf='olive', get_skin=True)
            print(degn_pose[0][0])
            #print(degn_trans)
            
            #loss = torch.abs(pred_silh_images[0, 0, :, :] - target_hg_silh)
            degn_trans = pred_trans.copy()
            degn_trans[0][0] = pred_trans[0][0] -float(torch.mean(keyp_3d[0,:24,0]).cpu().numpy())
            degn_trans[0][1] = pred_trans[0][1] -float(torch.mean(keyp_3d[0,:24,1]).cpu().numpy())

            degn_trans = torch.tensor(degn_trans, dtype=torch.float,device=device, requires_grad = True)
            optimizer = torch.optim.SGD([degn_trans], lr=5*1e-4,momentum=0.9)
            target_hg_silh = torch.tensor(cv2.resize((np.load(f'../GART/data/dog_data_official/{image_name}/images/{geo_list[i]:04d}.npy')),(256,256))).to(device)

            #original -> smal vertex를 얻기위해 smal을 한번 더 사용 . but 필요 없으니.
            loop = tqdm(range(100))
            for j in loop:
                optimizer.zero_grad()
                smal_verts, keyp_3d, _ = smal(beta=torch.from_numpy(pred_betas).to(device), betas_limbs=torch.from_numpy(pred_betas_limbs_notail).to(device), pose=torch.from_numpy(degn_pose).to(device), trans=degn_trans, keyp_conf='olive', get_skin=True)
                pred_silh_images, pred_keyp_raw = silh_renderer(vertices=smal_verts, points=keyp_3d, faces=faces_prep, focal_lengths=torch.from_numpy(pred_focal).to(device))
                loss = torch.mean(torch.abs(pred_silh_images[0, 0, :, :] - target_hg_silh)) #sum하니까 줄어들지 않는데 mean 하니까 줄어든다 왜그러지? 허허...
                print(f'{j}_loss = {loss}, degn_trans={degn_trans}')
            #loss.backward()
                #차이 확인하기.
                loss.backward(retain_graph=True)
                optimizer.step()
        
            #pred_keyp = pred_keyp_raw[:, :24, :]

            #img_silh = Image.fromarray(np.uint8(255*pred_silh_images[0, 0, :, :].detach().cpu().numpy())).convert('RGB')
            #at.append(img_silh)
            im =Image.open(f'../GART/data/dog_data_official/{image_name}/images/{geo_list[i]:04d}.png')
            im = im.resize((256,256))
            tt = np.array(im).astype(np.float32)
            tt = tt/255.0

            visualizations = silh_renderer.get_visualization_nograd(smal_verts, faces_prep, torch.from_numpy(pred_focal).to(device), color=0)
            pred_tex = visualizations[0, :, :, :].permute((1, 2, 0)).cpu().detach().numpy() / 256
            # out_path = root_out_path_details +  name + '_tex_pred_e' + format(i, '03d') + '.png'
            # plt.imsave(out_path, pred_tex)

            im_masked = cv2.addWeighted(tt,0.2,pred_tex,0.8,0)
            pred_tex_max = np.max(pred_tex, axis=2)
            im_masked[pred_tex_max<0.01, :] = tt[pred_tex_max<0.01, :]
            out_path = f'/home/user/gs/huggstudy/GART/data/dog_data_official/{image_name}/pred/{geo_list[i]:04d}.png'
            plt.imsave(out_path, im_masked)
            np.savez(f'/home/user/gs/huggstudy/GART/data/dog_data_official/{image_name}/pred/{geo_list[i]:04d}.npz',pred_betas=pred_betas, pred_betas_limbs=pred_betas_limbs_notail, pred_pose=degn_pose, pred_trans=degn_trans.detach().cpu().numpy(), pred_focal=pred_focal)


        # left rotation
        else:
            
            predz_left = dict(np.load(os.path.join(one_npz_path,f'{left_pose_start:04d}.npz')))


            pred_betas = predz_left['pred_betas']
            pred_betas_limbs = predz_left['pred_betas_limbs']
            pred_pose = predz_left['pred_pose']
            pred_trans = predz_left['pred_trans']
            pred_focal = predz_left['pred_focal']
            pred_betas_limbs_notail = pred_betas_limbs.copy()
            pred_betas_limbs_notail[0][2:4] =-0.300000


            b =pred_pose[0][0] @np.array([0,0,-1])
            y = np.array([0,1,0])
            rot5d = rot5deg2radmatrix(b,y,5)
            minusrot5d = np.linalg.inv(rot5d)
            if pointer==0:
                degn_pose = pred_pose.copy()
                
            degn_pose[0][0] =minusrot5d@degn_pose[0][0]
            smal_verts, keyp_3d, _ = smal(beta=torch.from_numpy(pred_betas).to(device), betas_limbs=torch.from_numpy(pred_betas_limbs_notail).to(device), pose=torch.from_numpy(degn_pose).to(device), trans=torch.from_numpy(pred_trans).float().to(device), keyp_conf='olive', get_skin=True)
            
            #print(degn_trans)
            
            #loss = torch.abs(pred_silh_images[0, 0, :, :] - target_hg_silh)
            degn_trans = pred_trans.copy()
            degn_trans[0][0] = pred_trans[0][0] -float(torch.mean(keyp_3d[0,:24,0]).cpu().numpy())
            degn_trans[0][1] = pred_trans[0][1] -float(torch.mean(keyp_3d[0,:24,1]).cpu().numpy())

            degn_trans = torch.tensor(degn_trans, dtype=torch.float,device=device, requires_grad = True)
            optimizer = torch.optim.SGD([degn_trans], lr=5*1e-4,momentum=0.9)
            target_hg_silh = torch.tensor(cv2.resize((np.load(f'../GART/data/dog_data_official/{image_name}/images/{max(geo_list)-pointer:04d}.npy')),(256,256))).to(device)

            #original -> smal vertex를 얻기위해 smal을 한번 더 사용 . but 필요 없으니.
            loop = tqdm(range(100))
            for j in loop:
                optimizer.zero_grad()
                smal_verts, keyp_3d, _ = smal(beta=torch.from_numpy(pred_betas).to(device), betas_limbs=torch.from_numpy(pred_betas_limbs_notail).to(device), pose=torch.from_numpy(degn_pose).to(device), trans=degn_trans, keyp_conf='olive', get_skin=True)
                pred_silh_images, pred_keyp_raw = silh_renderer(vertices=smal_verts, points=keyp_3d, faces=faces_prep, focal_lengths=torch.from_numpy(pred_focal).to(device))
                loss = torch.mean(torch.abs(pred_silh_images[0, 0, :, :] - target_hg_silh)) #sum하니까 줄어들지 않는데 mean 하니까 줄어든다 왜그러지? 허허...
                print(f'{j}_loss = {loss}, degn_trans={degn_trans}')
            #loss.backward()
                #차이 확인하기.
                loss.backward(retain_graph=True)
                optimizer.step()
        
            #pred_keyp = pred_keyp_raw[:, :24, :]

            #img_silh = Image.fromarray(np.uint8(255*pred_silh_images[0, 0, :, :].detach().cpu().numpy())).convert('RGB')
            #at.append(img_silh)
            im =Image.open(f'../GART/data/dog_data_official/{image_name}/images/{max(geo_list)-pointer:04d}.png')
            im = im.resize((256,256))
            tt = np.array(im).astype(np.float32)
            tt = tt/255.0

            visualizations = silh_renderer.get_visualization_nograd(smal_verts, faces_prep, torch.from_numpy(pred_focal).to(device), color=0)
            pred_tex = visualizations[0, :, :, :].permute((1, 2, 0)).cpu().detach().numpy() / 256
            # out_path = root_out_path_details +  name + '_tex_pred_e' + format(i, '03d') + '.png'
            # plt.imsave(out_path, pred_tex)

            im_masked = cv2.addWeighted(tt,0.2,pred_tex,0.8,0)
            pred_tex_max = np.max(pred_tex, axis=2)
            im_masked[pred_tex_max<0.01, :] = tt[pred_tex_max<0.01, :]
            out_path = f'/home/user/gs/huggstudy/GART/data/dog_data_official/{image_name}/pred/{max(geo_list)-pointer:04d}.png'
            plt.imsave(out_path, im_masked)
            np.savez(f'/home/user/gs/huggstudy/GART/data/dog_data_official/{image_name}/pred/{max(geo_list)-pointer:04d}.npz',pred_betas=pred_betas, pred_betas_limbs=pred_betas_limbs_notail, pred_pose=degn_pose, pred_trans=degn_trans.detach().cpu().numpy(), pred_focal=pred_focal)
            pointer +=1


    """
    for i in range(1,73):
        degn_pose[0][0] =rot5d@degn_pose[0][0]
        smal_verts, keyp_3d, _ = smal(beta=torch.from_numpy(pred_betas).to(device), betas_limbs=torch.from_numpy(pred_betas_limbs_notail).to(device), pose=torch.from_numpy(degn_pose).to(device), trans=torch.from_numpy(pred_trans).float().to(device), keyp_conf='olive', get_skin=True)
        
        #print(degn_trans)
        
        #loss = torch.abs(pred_silh_images[0, 0, :, :] - target_hg_silh)
        degn_trans = pred_trans.copy()
        degn_trans[0][0] = pred_trans[0][0] -float(torch.mean(keyp_3d[0,:24,0]).cpu().numpy())
        degn_trans[0][1] = pred_trans[0][1] -float(torch.mean(keyp_3d[0,:24,1]).cpu().numpy())

        degn_trans = torch.tensor(degn_trans, dtype=torch.float,device=device, requires_grad = True)
        optimizer = torch.optim.SGD([degn_trans], lr=5*1e-4,momentum=0.9)
        target_hg_silh = torch.tensor(cv2.resize((np.load(f'../GART/data/dog_data_official/{image_name}/images/{i:04d}.npy')),(256,256))).to(device)

        #original -> smal vertex를 얻기위해 smal을 한번 더 사용 . but 필요 없으니.
        loop = tqdm(range(100))
        for j in loop:
            optimizer.zero_grad()
            smal_verts, keyp_3d, _ = smal(beta=torch.from_numpy(pred_betas).to(device), betas_limbs=torch.from_numpy(pred_betas_limbs_notail).to(device), pose=torch.from_numpy(degn_pose).to(device), trans=degn_trans, keyp_conf='olive', get_skin=True)
            pred_silh_images, pred_keyp_raw = silh_renderer(vertices=smal_verts, points=keyp_3d, faces=faces_prep, focal_lengths=torch.from_numpy(pred_focal).to(device))
            loss = torch.mean(torch.abs(pred_silh_images[0, 0, :, :] - target_hg_silh)) #sum하니까 줄어들지 않는데 mean 하니까 줄어든다 왜그러지? 허허...
            print(f'{j}_loss = {loss}, degn_trans={degn_trans}')
           #loss.backward()
            #차이 확인하기.
            loss.backward(retain_graph=True)
            optimizer.step()
    
        #pred_keyp = pred_keyp_raw[:, :24, :]

        #img_silh = Image.fromarray(np.uint8(255*pred_silh_images[0, 0, :, :].detach().cpu().numpy())).convert('RGB')
        #at.append(img_silh)
        im =Image.open(f'../GART/data/dog_data_official/{image_name}/images/{i:04d}.png')
        im = im.resize((256,256))
        tt = np.array(im).astype(np.float32)
        tt = tt/255.0

        visualizations = silh_renderer.get_visualization_nograd(smal_verts, faces_prep, torch.from_numpy(pred_focal).to(device), color=0)
        pred_tex = visualizations[0, :, :, :].permute((1, 2, 0)).cpu().detach().numpy() / 256
        # out_path = root_out_path_details +  name + '_tex_pred_e' + format(i, '03d') + '.png'
        # plt.imsave(out_path, pred_tex)

        im_masked = cv2.addWeighted(tt,0.2,pred_tex,0.8,0)
        pred_tex_max = np.max(pred_tex, axis=2)
        im_masked[pred_tex_max<0.01, :] = tt[pred_tex_max<0.01, :]
        out_path = f'/home/user/gs/huggstudy/GART/data/dog_data_official/{image_name}/pred/{i:04d}.png'
        plt.imsave(out_path, im_masked)
        np.savez(f'/home/user/gs/huggstudy/GART/data/dog_data_official/{image_name}/pred/{i:04d}.npz',pred_betas=pred_betas, pred_betas_limbs=pred_betas_limbs_notail, pred_pose=degn_pose, pred_trans=degn_trans.detach().cpu().numpy(), pred_focal=pred_focal)
"""
    

if __name__ == "__main__":
    main()