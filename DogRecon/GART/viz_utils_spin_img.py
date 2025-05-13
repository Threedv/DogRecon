##############
from matplotlib import pyplot as plt
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix, matrix_to_quaternion
import imageio
import torch
from tqdm import tqdm
import numpy as np
import warnings, os, os.path as osp, shutil, sys
from transforms3d.euler import euler2mat
from lib_data.data_provider import RealDataOptimizablePoseProviderPose

from lib_gart.optim_utils import *
from lib_render.gauspl_renderer import render_cam_pcl
from lib_gart.model_utils import transform_mu_frame

from utils.misc import *
from utils.viz import viz_render

import pickle
from plyfile import PlyData, PlyElement
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
'''
@torch.no_grad()
def rot5deg2radmatrix(vector_a, vector_b, deg):
    dot_product = torch.dot(vector_a.squeeze(), vector_b.squeeze())
    magnitude_a = torch.norm(vector_a)
    magnitude_b = torch.norm(vector_b)
    
    cosine_angle = dot_product / (magnitude_a * magnitude_b)
    angle_radians = torch.acos(cosine_angle)
    
    angle_degrees = torch.rad2deg(angle_radians)
    ratio = angle_degrees / 90
    root_a = torch.deg2rad(torch.tensor(deg)) / (ratio**2 + (1-ratio)**2)**0.5
    
    rad5degaxis = torch.tensor([0.0, root_a * (1-ratio), root_a * ratio], device=vector_a.device)
    
    rotation_matrix = axis_angle_to_matrix(rad5degaxis.unsqueeze(0))
    
    return rotation_matrix
'''
'''
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

    return axis_angle_to_matrix(torch.from_numpy(rad5degaxis).float()) # numpy()
'''


@torch.no_grad()
def viz_spinning(
    model,
    pose,
    trans,
    H,
    W,
    K,
    save_path,
    time_index=None,
    n_spinning=10,
    model_mask=None,
    active_sph_order=0,
    bg_color=[1.0, 1.0, 1.0],
):
    device = pose.device
    mu, fr, s, o, sph, additional_ret = model(
        pose, trans, {"t": time_index}, active_sph_order=active_sph_order
    )
    if model_mask is not None:
        assert len(model_mask) == mu.shape[1]
        mu = mu[:, model_mask.bool()]
        fr = fr[:, model_mask.bool()]
        s = s[:, model_mask.bool()]
        o = o[:, model_mask.bool()]
        sph = sph[:, model_mask.bool()]

    viz_frames = []
    for vid in range(n_spinning):
        spin_R = (
            torch.from_numpy(euler2mat(0, 2 * np.pi * vid / n_spinning, 0, "sxyz"))
            .to(device)
            .float()
        )
        spin_t = mu.mean(1)[0]
        spin_t = (torch.eye(3).to(device) - spin_R) @ spin_t[:, None]
        spin_T = torch.eye(4).to(device)
        spin_T[:3, :3] = spin_R
        spin_T[:3, 3] = spin_t.squeeze(-1)
        viz_mu, viz_fr = transform_mu_frame(mu, fr, spin_T[None])

        render_pkg = render_cam_pcl(
            viz_mu[0],
            viz_fr[0],
            s[0],
            o[0],
            sph[0],
            H,
            W,
            K,
            False,
            active_sph_order,
            bg_color=bg_color,
        )
        viz_frame = (
            torch.clamp(render_pkg["rgb"], 0.0, 1.0)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        viz_frame = (viz_frame * 255).astype(np.uint8)
        viz_frames.append(viz_frame)
    imageio.mimsave(save_path, viz_frames)
    return


@torch.no_grad()
def viz_spinning_self_rotate(
    model,
    base_R,
    pose,
    trans,
    H,
    W,
    K,
    save_path,
    time_index=None,
    n_spinning=10,
    model_mask=None,
    active_sph_order=0,
    bg_color=[1.0, 1.0, 1.0],
):
    device = pose.device
    viz_frames = []
    # base_R = base_R.detach().cpu().numpy()
    first_R = axis_angle_to_matrix(pose[:, 0])[0].detach().cpu().numpy()
    for vid in range(n_spinning):
        rotation = euler2mat(0.0, 2 * np.pi * vid / n_spinning, 0.0, "sxyz")
        rotation = torch.from_numpy(first_R @ rotation).float().to(device)
        pose[:, 0] = matrix_to_axis_angle(rotation[None])[0]

        mu, fr, s, o, sph, additional_ret = model(
            pose, trans, {"t": time_index}, active_sph_order=active_sph_order
        )
        if model_mask is not None:
            assert len(model_mask) == mu.shape[1]
            mu = mu[:, model_mask.bool()]
            fr = fr[:, model_mask.bool()]
            s = s[:, model_mask.bool()]
            o = o[:, model_mask.bool()]
            sph = sph[:, model_mask.bool()]

        render_pkg = render_cam_pcl(
            mu[0],
            fr[0],
            s[0],
            o[0],
            sph[0],
            H,
            W,
            K,
            False,
            active_sph_order,
            bg_color=bg_color,
        )
        viz_frame = (
            torch.clamp(render_pkg["rgb"], 0.0, 1.0)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        viz_frame = (viz_frame * 255).astype(np.uint8)
        viz_frames.append(viz_frame)
    imageio.mimsave(save_path, viz_frames)
    return


@torch.no_grad()
def viz_human_all(
    solver,
    data_provider: RealDataOptimizablePoseProviderPose = None,
    ckpt_dir=None,
    training_skip=1,
    n_spinning=40,
    novel_pose_dir="novel_poses",
    novel_skip=2,
    model=None,
    model_mask=None,
    viz_name="",
    export_mesh_flag=False,  # remove this from release version
):
    if model is None:
        model = solver.load_saved_model(ckpt_dir)
    model.eval()

    viz_dir = osp.join(solver.log_dir, f"{viz_name}_human_viz")
    os.makedirs(viz_dir, exist_ok=True)

    active_sph_order = int(model.max_sph_order)

    if data_provider is not None:
        # if ckpt_dir is None:
        #     ckpt_dir = solver.log_dir
        # pose_path = osp.join(ckpt_dir, "pose.pth")
        pose_base_list = data_provider.pose_base_list
        pose_rest_list = data_provider.pose_rest_list
        global_trans_list = data_provider.global_trans_list
        pose_list = torch.cat([pose_base_list, pose_rest_list], 1)
        pose_list, global_trans_list = pose_list.to(
            solver.device
        ), global_trans_list.to(solver.device)
        rgb_list = data_provider.rgb_list
        mask_list = data_provider.mask_list
        K_list = data_provider.K_list
        H, W = rgb_list.shape[1:3]
    else:
        H, W = 512, 512
        K_list = [torch.from_numpy(fov2K(45, H, W)).float().to(solver.device)]
        global_trans_list = torch.zeros(1, 3).to(solver.device)
        global_trans_list[0, -1] = 3.0

    # viz training
    if data_provider is not None:
        print("Viz training...")
        viz_frames = []
        for t in range(len(pose_list)):
            if t % training_skip != 0:
                continue
            pose = pose_list[t][None]
            K = K_list[t]
            trans = global_trans_list[t][None]
            time_index = torch.Tensor([t]).long().to(solver.device)
            mu, fr, s, o, sph, _ = model(
                pose,
                trans,
                {"t": time_index},  # use time_index from training set
                active_sph_order=active_sph_order,
            )
            if model_mask is not None:
                assert len(model_mask) == mu.shape[1]
                mu = mu[:, model_mask.bool()]
                fr = fr[:, model_mask.bool()]
                s = s[:, model_mask.bool()]
                o = o[:, model_mask.bool()]
                sph = sph[:, model_mask.bool()]
            render_pkg = render_cam_pcl(
                mu[0],
                fr[0],
                s[0],
                o[0],
                sph[0],
                H,
                W,
                K,
                False,
                active_sph_order,
                bg_color=getattr(solver, "DEFAULT_BG", [1.0, 1.0, 1.0]),
            )
            viz_frame = viz_render(rgb_list[t], mask_list[t], render_pkg)
            viz_frames.append(viz_frame)
        imageio.mimsave(f"{viz_dir}/training.gif", viz_frames)

    # viz static spinning
    print("Viz spinning...")
    can_pose = model.template.canonical_pose.detach()
    viz_base_R_opencv = np.asarray(euler2mat(np.pi, 0, 0, "sxyz"))
    viz_base_R_opencv = torch.from_numpy(viz_base_R_opencv).float()
    can_pose[0] = viz_base_R_opencv.to(can_pose.device)
    can_pose = matrix_to_axis_angle(can_pose)[None]
    dapose = torch.from_numpy(np.zeros((1, 24, 3))).float().to(solver.device)
    dapose[:, 1, -1] = np.pi / 4
    dapose[:, 2, -1] = -np.pi / 4
    dapose[:, 0] = matrix_to_axis_angle(solver.viz_base_R[None])[0]
    tpose = torch.from_numpy(np.zeros((1, 24, 3))).float().to(solver.device)
    tpose[:, 0] = matrix_to_axis_angle(solver.viz_base_R[None])[0]
    to_viz = {"cano-pose": can_pose, "t-pose": tpose, "da-pose": dapose}
    if data_provider is not None:
        to_viz["first-frame"] = pose_list[0][None]

    for name, pose in to_viz.items():
        print(f"Viz novel {name}...")
        # if export_mesh_flag:
        #     from lib_marchingcubes.gaumesh_utils import MeshExtractor
        #     # also extract a mesh
        #     mesh = solver.extract_mesh(model, pose)
        #     mesh.export(f"{viz_dir}/mc_{name}.obj", "obj")

        # # for making figures, the rotation is in another way
        # viz_spinning_self_rotate(
        #     model,
        #     solver.viz_base_R.detach(),
        #     pose,
        #     global_trans_list[0][None],
        #     H,
        #     W,
        #     K_list[0],
        #     f"{viz_dir}/{name}_selfrotate.gif",
        #     time_index=None,  # if set to None and use t, the add_bone will hand this
        #     n_spinning=n_spinning,
        #     active_sph_order=model.max_sph_order,
        # )
        viz_spinning(
            model,
            pose,
            global_trans_list[0][None],
            H,
            W,
            K_list[0],
            f"{viz_dir}/{name}.gif",
            time_index=None,  # if set to None and use t, the add_bone will hand this
            n_spinning=n_spinning,
            active_sph_order=model.max_sph_order,
            bg_color=getattr(solver, "DEFAULT_BG", [1.0, 1.0, 1.0]),
        )

    # viz novel pose dynamic spinning
    print("Viz novel seq...")
    novel_pose_names = [
        f[:-4] for f in os.listdir(novel_pose_dir) if f.endswith(".npy")
    ]
    seq_viz_todo = {}
    for name in novel_pose_names:
        novel_pose_fn = osp.join(novel_pose_dir, f"{name}.npy")
        novel_poses = np.load(novel_pose_fn, allow_pickle=True)
        novel_poses = novel_poses[::novel_skip]
        N_frames = len(novel_poses)
        novel_poses = torch.from_numpy(novel_poses).float().to(solver.device)
        novel_poses = novel_poses.reshape(N_frames, 24, 3)

        seq_viz_todo[name] = (novel_poses, N_frames)
    if data_provider is not None:
        seq_viz_todo["training"] = [pose_list, len(pose_list)]

    for name, (novel_poses, N_frames) in seq_viz_todo.items():
        base_R = solver.viz_base_R.detach().cpu().numpy()
        viz_frames = []
        K = K_list[0]
        for vid in range(N_frames):
            pose = novel_poses[vid][None]
            # pose = novel_poses[0][None] # debug
            rotation = euler2mat(2 * np.pi * vid / N_frames, 0.0, 0.0, "syxz")
            rotation = torch.from_numpy(rotation @ base_R).float().to(solver.device)
            pose[:, 0] = matrix_to_axis_angle(rotation[None])[0]
            trans = global_trans_list[0][None]
            mu, fr, s, o, sph, _ = model(
                pose,
                trans,
                # not pass in {}, so t is auto none
                additional_dict={},
                active_sph_order=active_sph_order,
            )
            if model_mask is not None:
                assert len(model_mask) == mu.shape[1]
                mu = mu[:, model_mask.bool()]
                fr = fr[:, model_mask.bool()]
                s = s[:, model_mask.bool()]
                o = o[:, model_mask.bool()]
                sph = sph[:, model_mask.bool()]
            render_pkg = render_cam_pcl(
                mu[0],
                fr[0],
                s[0],
                o[0],
                sph[0],
                H,
                W,
                K,
                False,
                active_sph_order,
                bg_color=getattr(solver, "DEFAULT_BG", [1.0, 1.0, 1.0]),
                # bg_color=[1.0, 1.0, 1.0],  # ! use white bg for viz
            )
            viz_frame = (
                torch.clamp(render_pkg["rgb"], 0.0, 1.0)
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
            )
            viz_frame = (viz_frame * 255).astype(np.uint8)
            viz_frames.append(viz_frame)
        imageio.mimsave(f"{viz_dir}/novel_pose_{name}.gif", viz_frames)
    return


def viz_dog_spin(
    model, pose, trans, H, W, K, save_path, n_spinning=10, device="cuda:0"
):
    BASE_R = np.asarray(euler2mat(np.pi / 2.0, 0, np.pi, "rxyz"))
    novel_view_pitch = 0.15  # or 0

    viz_frames = []
    angles = np.linspace(0.65, 0.90, n_spinning)
    angles = np.concatenate([angles, angles[::-1]])

    for angle in angles:
        rotation = euler2mat(2 * np.pi * angle, novel_view_pitch, 0.0, "syxz")
        rotation = torch.from_numpy(rotation @ BASE_R).float().to(device)
        pose[:, :3] = matrix_to_axis_angle(rotation[None])[0]

        mu, fr, s, o, sph, additional_ret = model(
            pose, trans, {}, active_sph_order=model.max_sph_order
        )

        render_pkg = render_cam_pcl(
            mu[0],
            fr[0],
            s[0],
            o[0],
            sph[0],
            H,
            W,
            K,
            False,
            model.max_sph_order,
            bg_color=[1.0, 1.0, 1.0],
        )

        viz_frame = (
            torch.clamp(render_pkg["rgb"], 0.0, 1.0)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        viz_frame = (viz_frame * 255).astype(np.uint8)
        viz_frames.append(viz_frame)

    imageio.mimsave(save_path, viz_frames, fps=30)

    return

def viz_dog_spin2(model, pose, trans, H, W, K, save_path, n_spinning=20):

    device = trans.device
    BASE_R = np.asarray(euler2mat(np.pi / 2.0, 0, np.pi, "rxyz"))
    trans = trans.detach().clone()

    frames_dir = os.path.join(save_path, 'viz_frames_spin3')
    os.makedirs(frames_dir, exist_ok=True)

    viz_frames = []
    i=0
    for vid in range(n_spinning):
        i=i+1
        rotation = euler2mat(
            2 * np.pi * vid / n_spinning, 0.0, 0.0, "syxz"
        )
        rotation = torch.from_numpy(rotation @ BASE_R).float().to(device)
        pose[:, :3] = matrix_to_axis_angle(rotation[None])[0]
        mu, fr, s, o, sph, additional_ret = model(pose, trans, active_sph_order=model.max_sph_order)
        render_pkg = render_cam_pcl(
            mu[0],
            fr[0],
            s[0],
            o[0],
            sph[0],
            H,
            W,
            K,
            False,
            model.max_sph_order,
            bg_color=[1.0, 1.0, 1.0],
        )

        viz_frame = (
            torch.clamp(render_pkg["rgb"], 0.0, 1.0)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        viz_frame = (viz_frame * 255).astype(np.uint8)
        
        frame_path = os.path.join(frames_dir, f'frame_{i:03d}_test.png')
        imageio.imwrite(frame_path, viz_frame)


def viz_dog_spin3(model, pose, trans, H, W, K, save_path, n_spinning=10, device="cuda:0"):
    #b = pose[0][0] @ np.array([0,0,-1])
    print(f'pose.shape:{pose.shape}')
    print(f'np.array([0,0,-1]).shape:{np.array([0,0,-1]).shape}')
    BASE_R = np.asarray(euler2mat(np.pi / 2.0, 0, np.pi, "rxyz"))
    print(f'BASE_R.shape:{BASE_R.shape}')

    ##########
    limb = pose[:, -7:]  #(1,7)
    pose = pose[:, :-7].reshape(-1, 35, 3) # 원래 pose의 shape는 (1, 112) - > (1,35,3)가 된다.
    print(f'pose.shape:{pose.shape}')

    root_pose = pose[:,0:1,:].squeeze(0) # 이렇게 하면, (1,3)이 될 것!
    print(f'root_pose.shape:{root_pose.shape}')
    root_pose_rot_matrix = axis_angle_to_matrix(root_pose) # 이렇게 하면, (1,3,3)
    print(f'root_pose_rot_matrix.shape:{root_pose_rot_matrix.shape}')

    BASE_R_tensor = torch.from_numpy(BASE_R).float().to(root_pose_rot_matrix.device)
    root_pose_rot_matrix = BASE_R_tensor @ root_pose_rot_matrix
    print(f'root_pose_rot_matrix.shape:{root_pose_rot_matrix.shape}')

    #pose_rot_matrix = axis_angle_to_matrix(pose) # 이렇게 하면, (1,35,3,3)

    #rotation matrix 구하기
    #b = root_pose_rot_matrix.cpu().numpy() @ np.array([0,0,-1])
    #y = np.array([0,1,0])
    #rot5d = rot5deg2radmatrix(b,y,5)

    frames_dir = os.path.join(save_path, 'viz_frames_spin3')
    os.makedirs(frames_dir, exist_ok=True)

    for i in range(73):
        #if i>0:
            #root_pose_rot_matrix = rot5d.to(root_pose_rot_matrix.device) @ root_pose_rot_matrix # (1,3,3)
            #root_pose_axis_angle = matrix_to_axis_angle(root_pose_rot_matrix) #(1,3)

            #pose[:, 0, :] = root_pose_axis_angle
        root_pose_axis_angle = matrix_to_axis_angle(root_pose_rot_matrix).unsqueeze(0) #(1,1,3)
        print(f'root_pose_axis_angle.shape:{root_pose_axis_angle.shape}')
        print(f'pose[:, 0:1, :].shape:{pose[:, 0:1, :].shape}')
        pose[:, 0:1, :] = root_pose_axis_angle

        pose = pose.reshape(1, -1)
        print(f'pose.shape:{pose.shape}')
        print(f'limb.shape:{limb.shape}')
        pose2 = torch.cat((pose, limb), dim=1) #(1,112)
        print(f'pose2.shape!!!!!!!:{pose2.shape}')
    
        mu, fr, s, o, sph, additional_ret = model(
            pose2, trans, {}, active_sph_order=model.max_sph_order
        )

        render_pkg = render_cam_pcl(
            mu[0],
            fr[0],
            s[0],
            o[0],
            sph[0],
            H,
            W,
            K,
            False,
            model.max_sph_order,
            bg_color=[1.0, 1.0, 1.0],
        )

        viz_frame = (
            torch.clamp(render_pkg["rgb"], 0.0, 1.0)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        viz_frame = (viz_frame * 255).astype(np.uint8)
        
        frame_path = os.path.join(frames_dir, f'frame_{i:03d}_refine.png')
        imageio.imwrite(frame_path, viz_frame)

    return
'''
def process_animation(model, animation, animation_name, limb, trans, H, W, K, device):
    frames = []
    gart_dict = []
    for pose in tqdm(animation, desc=f'Processing {animation_name}') :
        pose = pose.reshape(1, -1)
        pose = torch.cat([pose, limb], dim=1)

        mu, fr, s, o, sph, _ = model(
            pose, trans, {}, active_sph_order=model.max_sph_order
        )

        render_pkg = render_cam_pcl(
            mu[0], fr[0], s[0], o[0], sph[0], H, W, K, False, model.max_sph_order, bg_color=[1.0, 1.0, 1.0]
        )

        viz_frame = torch.clamp(render_pkg["rgb"], 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
        viz_frame = (viz_frame * 255).astype(np.uint8)
        frames.append(viz_frame)

        mu = mu[0].squeeze(0)  # (12240, 3)
        o = o[0].squeeze(0)  # (12240, 1)
        s = s[0].squeeze(0)  # (12240, 3)
        fr = fr[0].squeeze(0) #12240,3,3
        sph = sph[0].squeeze(0)
        
        fr = matrix_to_quaternion(fr)
        gart_key = {'mean3D': mu.detach().cpu().numpy(), 'rotations': fr.detach().cpu().numpy(), 'scales': s.detach().cpu().numpy(), 'opacity': o.detach().cpu().numpy(), 'shs':sph.detach().cpu().numpy()}
        
        gart_dict.append(gart_key)

    return frames, gart_dict
    '''
def process_animation(model, animation, animation_name, limb, trans, H, W, K, save_path, device):
    frames_dir = os.path.join(save_path, animation_name)
    os.makedirs(frames_dir, exist_ok=True)

    gart_dict = []
    for i, pose in enumerate(tqdm(animation, desc=f'Processing {animation_name}')):
        pose = pose.reshape(1, -1)
        pose = torch.cat([pose, limb], dim=1)

        mu, fr, s, o, sph, _ = model(
            pose, trans, {}, active_sph_order=model.max_sph_order
        )

        render_pkg = render_cam_pcl(
            mu[0], fr[0], s[0], o[0], sph[0], H, W, K, False, model.max_sph_order, bg_color=[1.0, 1.0, 1.0]
        )

        viz_frame = torch.clamp(render_pkg["rgb"], 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
        viz_frame = (viz_frame * 255).astype(np.uint8)

        frame_path = os.path.join(frames_dir, f'frame_{i:03d}.png')
        imageio.imwrite(frame_path, viz_frame)

    return gart_dict

def viz_dog_animation(model, animations, animation_names, limb, trans, H, W, K, save_paths, seq_name, device="cuda:0"):
    save_dir = '/home/user/gs/huggstudy/GART/ijcv_fig6_ours_test'
    
    for animation, save_path, animation_name in zip(animations, save_paths, animation_names):
        print(f'animation_name:{animation_name}')
        gart_dict = process_animation(model, animation, animation_name, limb, trans, H, W, K, save_path, device)

        print(f'{animation_name} are successfully saved in {save_path}.')


'''
def viz_dog_animation(model, animations, animation_names, limb, trans, H, W, K, save_paths, seq_name, fps=90, device="cuda:0"):
    save_dir = '/root/dev/oneshotgart/GART/gart_output'
    
    for animation, save_path, animation_name in zip(animations, save_paths, animation_names):
        print(f'animation_name:{animation_name}')
        frames, gart_dict = process_animation(model, animation, animation_name, limb, trans, H, W, K, device)
        imageio.mimsave(save_path, frames, fps=fps)

        print(f'{os.path.basename(save_path)} are successfully saved.')
'''


@torch.no_grad()
def viz_dog_all(solver, data_provider, model=None, ckpt_dir=None, viz_name="", seq_name=""):
    if model is None:
        model = solver.load_saved_model(ckpt_dir)
    model.eval()
    viz_dir = osp.join(solver.log_dir, f"{seq_name}_dog_viz")
    os.makedirs(viz_dir, exist_ok=True)

    
    viz_pose = (
        torch.cat([data_provider.pose_base_list, data_provider.pose_rest_list], 1)
        .detach()
        .clone()
    )

    #root_pose_file = np.load("/root/dev/oneshotgart/GART/data/dog_data_official/d1_flip/pred/0000.npz")
    #root_pose_np = np.load("/root/dev/oneshotgart/GART/data/dog_data_official/d1_flip/pred/0000.npz")["pred_pose"] 


    viz_pose = (
        torch.cat([data_provider.pose_base_list, data_provider.pose_rest_list], 1)
        .detach()
        .clone()
    )
    viz_pose = torch.mean(viz_pose, dim=0, keepdim=True)
    pose = viz_pose[:, :-7].reshape(-1, 35, 3)
    limb = viz_pose[:, -7:]     
    trans = torch.tensor([[0.3, -0.3, 25.0]], device="cuda:0")
    
    print(f'seq_name:{seq_name}')
    aroot4 = osp.join(osp.dirname(__file__), f"/home/user/gs/huggstudy/GART/data/dog_data_official/{seq_name}/pred")
    print(aroot4)
    window_d61 = list(range(0, 144))
    files_d61 = [f"{aroot4}/{m:04d}.npz" for m in window_d61]
    pose_list_d61 = [dict(np.load(file_d))["pred_pose"] for file_d in files_d61]
    pose_list_d61 = np.concatenate(pose_list_d61)
    animation6 = matrix_to_axis_angle(torch.from_numpy(pose_list_d61)).to(solver.device)

    animation6[:, [32, 33, 34]] = pose[:, [32, 33, 34]]
    animations = [animation6]
    animation_names = [f'animation_spin_{seq_name}']
    save_path6 = osp.join(viz_dir, f"spin_images_{seq_name}")
    save_paths = [save_path6]

    viz_dog_animation(
        model.to("cuda"),
        animations,
        animation_names,
        limb,
        trans,
        data_provider.H,
        data_provider.W,
        data_provider.K_list[0],
        save_paths,
        seq_name,
        #save_path3=osp.join(viz_dir, f"animation_{seq_name}_walk_front.gif"),
        device="cuda:0"
    )

    return
