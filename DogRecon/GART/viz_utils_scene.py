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

from PIL import Image

from plyfile import PlyData, PlyElement
import numpy as np


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

    viz_frames = []
    for vid in range(n_spinning):
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
            torch.clamp(render_pkg["rgb"], 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
        )
        viz_frame = (viz_frame * 255).astype(np.uint8)
        viz_frames.append(viz_frame)
    imageio.mimsave(save_path, viz_frames)
    return

def process_animation(model, animation, animation_name, frames_dir, limb, trans, H, W, K, device):
    frames = []
    gart_dict = []
    i = -1
    #for pose in tqdm(animation, desc=f'Processing {animation_name}') :
    for i, pose in enumerate(tqdm(animation, desc=f'Processing {animation_name}')) :
        i += 1
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

        frame_path = os.path.join(frames_dir, f'frame_{i:04d}.png')
        imageio.imwrite(frame_path, viz_frame)

        mu = mu[0].squeeze(0)  # (12240, 3)
        o = o[0].squeeze(0)  # (12240, 1)
        s = s[0].squeeze(0)  # (12240, 3)
        fr = fr[0].squeeze(0) #12240,3,3
        sph = sph[0].squeeze(0)
        
        fr = matrix_to_quaternion(fr)
        gart_key = {'mean3D': mu.detach().cpu().numpy(), 'rotations': fr.detach().cpu().numpy(), 'scales': s.detach().cpu().numpy(), 'opacity': o.detach().cpu().numpy(), 'shs':sph.detach().cpu().numpy()}
        
        gart_dict.append(gart_key)

        ply_path = '/home/user/gs/huggstudy/GART/dh_dogrecon/test.ply'

        vertices = np.zeros(mu.shape[0], dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('alpha', 'u1')
        ])
        mu_np = mu.detach().cpu().numpy()
        o_np = o.detach().cpu().numpy().reshape(-1)
        alpha_np = (np.clip(o_np, 0, 1) * 255).astype(np.uint8)

        vertices['x'] = mu_np[:, 0]
        vertices['y'] = mu_np[:, 1]
        vertices['z'] = mu_np[:, 2]
        vertices['red'] = 255
        vertices['green'] = 255
        vertices['blue'] = 255
        vertices['alpha'] = alpha_np

        PlyData([PlyElement.describe(vertices, 'vertex')]).write(ply_path)

        if i == 1:
            break

    return frames, gart_dict


def viz_dog_animation(model, animations, animation_names, save_path_animation_img, limb, trans, H, W, K, save_paths,seq_name, fps=90, device="cuda:0"):
    save_dir = '/home/user/gs/huggstudy/GART/dh_dogrecon/test'
    
    count = 0
    for animation, save_path, animation_name in zip(animations, save_paths, animation_names):
        
        frames_dir = os.path.join(save_path_animation_img, f'{seq_name}_{animation_name}_frame_image')
        os.makedirs(frames_dir, exist_ok=True)
        
        print(f'animation_name:{animation_name}')
        frames, gart_dict = process_animation(model, animation, animation_name, frames_dir, limb, trans, H, W, K, device)
        count += 1
        if count == 1:
            print('ddddddddddd')
            break
        imageio.mimsave(save_path, frames, fps=fps)

        pickle_dir = os.path.join(save_dir, seq_name)
        os.makedirs(pickle_dir, exist_ok=True)
        
        pickle_path = f'{pickle_dir}/{animation_name}_dict.pickle'  # Use animation name for pickle file
        with open(pickle_path, 'wb') as f:
            pickle.dump(gart_dict, f, protocol=4)

        print(f'{os.path.basename(save_path)} and {os.path.basename(pickle_path)} are successfully saved.')


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
 
    viz_pose = torch.mean(viz_pose, dim=0, keepdim=True)   # use mean pose for viz  
    limb = viz_pose[:, -7:]                                
    pose = viz_pose[:, :-7].reshape(-1, 35, 3)
    pose[:, :-3] = 0  # exclude ears and mouth poses


    viz_pose = torch.concat([pose.reshape(1, -1), limb], dim=1)
    viz_trans = torch.tensor([[0.0, -0.3, 25.0]], device="cuda:0")

 
    viz_dog_spin(
        model.to("cuda"),
        viz_pose,
        viz_trans,
        data_provider.H,
        data_provider.W,
        data_provider.K_list[0],
        save_path=osp.join(viz_dir, f'spin_{seq_name}.gif'),
        n_spinning=42,
    )
    viz_dog_spin2(
        model.to("cuda"),
        viz_pose,
        viz_trans,
        data_provider.H,
        data_provider.W,
        data_provider.K_list[0],
        save_path=osp.join(viz_dir, f'spin2_{seq_name}.gif'),
        n_spinning=20,
    )

    ######################################################################
    # Dataset pose seq
    viz_pose = (
        torch.cat([data_provider.pose_base_list, data_provider.pose_rest_list], 1)
        .detach()
        .clone()
    )
    viz_pose = torch.mean(viz_pose, dim=0, keepdim=True)
    pose = viz_pose[:, :-7].reshape(-1, 35, 3)
    limb = viz_pose[:, -7:]

    # Animation -> 첫번째가 walking, 두번째가 drone
    aroot = osp.join(osp.dirname(__file__), "novel_poses/husky")
    aroot2 = osp.join(osp.dirname(__file__), "novel_poses/data/dog_data_official/animation_test_drone/pred")
    aroot3 = osp.join(osp.dirname(__file__), "novel_poses/data/dog_data_official/animation_test_dance/pred")
    #window = list(range(0, 146))  # Run
    #window = list(range(350, 440))  # Run


    #350~366 써서 반복하자!!!
    #357~366 써서 반복하자!!!
    # drone은 91~109!
    window_gart_original = list(range(350, 450))  # Run


    window_front_view = list(range(439, 440))

    window = list(range(359, 366)) # 7
    window2 = list(range(91, 109))  # 17
    window_dance = list(range(43,55)) # 12
    
    trans = torch.tensor([[0.3, -0.3, 25.0]], device="cuda:0")
    files = [f"{aroot}/{i:04d}.npz" for i in window]
    files2 = [f"{aroot2}/{j:04d}.npz" for j in window2]
    files3 = [f"{aroot}/{u:04d}.npz" for u in window_front_view]
    files_dance = [f"{aroot3}/{g:04d}.npz" for g in window_dance]
    files_gart_original = [f"{aroot}/{w:04d}.npz" for w in window_gart_original]

    pose_front_list = [dict(np.load(file3))["pred_pose"] for file3 in files3]
    pose_front_array = np.stack(pose_front_list, axis=0)
    #pose_front_tensor = torch.tensor(pose_front_list, dtype=torch.float32, device='cuda')

    pose_list = [dict(np.load(file))["pred_pose"] for file in files]
    pose_list = np.concatenate(pose_list)

    pose_list2 = [dict(np.load(file2))["pred_pose"] for file2 in files2]
    pose_list2 = np.concatenate(pose_list2)
    #print(f'pose_list.shape:{pose_list.shape}') #(7,35,3,3)
    pose_list_dance = [dict(np.load(file_dance))["pred_pose"] for file_dance in files_dance]
    pose_list_dance = np.concatenate(pose_list_dance)

    pose_list_gart = [dict(np.load(file_gart_original))["pred_pose"] for file_gart_original in files_gart_original]
    pose_list_gart = np.concatenate(pose_list_gart)

    ################################
    # dance
    poses_forward_dance = pose_list_dance  # 1~12 12개 
    #poses_backward = pose_list[::-1]
    poses_backward_dance = pose_list_dance[-2::-1] # 11~1 11개
    poses_forward_2_dance = pose_list_dance[1:] # 2~12 11개

    new_pose_list_dance = np.concatenate((poses_forward_dance, poses_backward_dance, poses_forward_2_dance, poses_backward_dance, poses_forward_2_dance,poses_backward_dance, poses_forward_2_dance,poses_backward_dance, poses_forward_2_dance, poses_backward_dance)) # 12+11*9 = 111

    ################################

    poses_forward = pose_list  # 1~7 7개 
    #poses_backward = pose_list[::-1]
    poses_backward = pose_list[-2::-1] # 6~1 6개
    poses_forward_2 = pose_list[1:] # 2~7 6개

    poses_forward_d = pose_list2  # 1~7 7개 
    #poses_backward = pose_list[::-1]
    poses_backward_d = pose_list2[-2::-1] # 6~1 6개
    poses_forward_2_d = pose_list2[1:] # 2~7 6개

    #print(len(poses_backward),len(poses_forward_2))

    n_pose_list = np.concatenate((poses_forward, poses_backward, poses_forward_2, poses_backward, poses_forward_2,poses_backward, poses_forward_2,poses_backward, poses_forward_2,poses_backward, poses_forward_2, poses_backward, poses_forward_2, poses_backward, poses_forward_2,poses_backward, poses_forward_2,poses_backward)) # 7+6*17 = 109
    n_pose_list2 = np.concatenate((poses_forward_d, poses_backward_d, poses_forward_2_d, poses_backward_d, poses_forward_2_d,poses_backward_d, poses_forward_2_d)) # 9+17*6 = 111
    #print(f'n_pose_list.shape:{n_pose_list.shape}')  #(109,35,3,3)

    #############################################
    
    pose_smal_list = np.zeros((21, 35, 3), dtype = np.float32)

    start_end_values2 = {
        7: (-0.35, 0.5),
        11: (0.53, -0.28),
        17: (0.5, -0.35),
        21: (-0.28, 0.53)
    }

    for joint2, (start_angle2, end_angle2) in start_end_values2.items():
        interpolated_angles2 = np.linspace(start_angle2, end_angle2, 21)
        for i in range(21):
            pose_smal_list[i, joint2, :] = [0, interpolated_angles2[i], 0]

    poses_forward_smal = pose_smal_list  # 1~7 7개 
    #poses_backward = pose_list[::-1]
    poses_backward_smal = pose_smal_list[-2::-1] # 6~1 6개
    poses_forward_2_smal = pose_smal_list[1:] # 2~7 6개

    #print(len(poses_backward),len(poses_forward_2))

    new_pose_smal_list = np.concatenate((poses_forward_smal, poses_backward_smal, poses_forward_2_smal,  poses_backward_smal, poses_forward_2_smal, poses_backward_smal, poses_forward_2_smal, poses_backward_smal, poses_forward_2_smal, poses_backward_smal, poses_forward_2_smal,  poses_backward_smal, poses_forward_2_smal,  poses_backward_smal, poses_forward_2_smal, poses_backward_smal, poses_forward_2_smal,poses_backward_smal)) # 9+6*17 = 111
    
    print(f'new_pose_smal_list.shape= {new_pose_smal_list.shape}') #()
    

    #############################################
    '''
    def axis_angle_to_rotation_matrix(axis_angle):
        theta = np.linalg.norm(axis_angle)
        if theta > 0.0001:
            axis = axis_angle / theta
            K = np.array([[0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]])
            R2 = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
        else:
            R2 = np.eye(3)
        return R2
    '''

    final_pose_list = np.zeros((21, 35, 3), dtype = np.float32)

    start_end_values = {
        4: (-0.06, 0.0),
        7: (-0.35, 0.5),
        8: (0.0, 0.0),
        9: (0.0, 0.0),
        10: (0.0, 0.0),
        11: (0.53, -0.28),
        12: (0.0, 0.0),
        13: (0.0, 0.0),
        14: (0.0, 0.0),
        15: (0.31, 0.0),
        15: (-0.22, 0.0),
        17: (0.5, -0.35),
        18: (0.0, 0.0),
        19: (0.0, 0.0),
        20: (0.0, 0.0),
        21: (-0.28, 0.53),
        22: (0.0, 0.0),
        23: (0.0, 0.0),
        24: (0.0, 0.0)
    }

    for joint, (start_angle, end_angle) in start_end_values.items():
        interpolated_angles = np.linspace(start_angle, end_angle, 21)

        if joint == 4:
            interpolated_angles1 = np.linspace(start_angle, end_angle, 11)
            interpolated_angles2 = np.linspace(0.0, 0.06, 10)
            interpolated_angles = np.concatenate((interpolated_angles1, interpolated_angles2))

        if joint == 15:
            interpolated_angles1 = np.linspace(start_angle, end_angle, 11)
            interpolated_angles2 = np.linspace(0.0, -0.31, 10)
            interpolated_angles = np.concatenate((interpolated_angles1, interpolated_angles2))

        if joint == 16:
            interpolated_angles1 = np.linspace(start_angle, end_angle, 10)
            interpolated_angles2 = np.linspace(0.0, 0.22, 11)
            interpolated_angles = np.concatenate((interpolated_angles1, interpolated_angles2))
        
        #print(f'interpolated_angles.shape={interpolated_angles.shape},{interpolated_angles}') interpolated_angles.shape=(7,),[0. 0. 0. 0. 0. 0. 0.]
        for i in range(21):
            # Axis-angle로 변환
            axis_angle = [0, interpolated_angles[i], 0]
            if joint ==16:
                axis_angle = [interpolated_angles[i], 0, 0]

            # 회전 행렬로 변환
            #R2 = axis_angle_to_rotation_matrix(axis_angle)
            #print(f'R2.shape={R2.shape},{R2}')
            # pose_list 업데이트
            final_pose_list[i, joint, :] = axis_angle
           # print(f'final_pose_list.shape={final_pose_list.shape},{final_pose_list}')


    poses_forward_final = final_pose_list  # 1~7 7개 
    #poses_backward = pose_list[::-1]
    poses_backward_final = final_pose_list[-2::-1] # 6~1 6개
    poses_forward_2_final = final_pose_list[1:] # 2~7 6개

    #print(len(poses_backward),len(poses_forward_2))

    n_final_pose_list = np.concatenate((poses_forward_final, poses_backward_final, poses_forward_2_final, poses_backward_final, poses_forward_2_final,poses_backward_final, poses_forward_2_final,poses_backward_final, poses_forward_2_final,poses_backward_final, poses_forward_2_final, poses_backward_final, poses_forward_2_final, poses_backward_final, poses_forward_2_final,poses_backward_final, poses_forward_2_final,poses_backward_final)) # 9+6*17 = 111
    print(f'n_final_pose_list.shape:{n_final_pose_list.shape}')
    
    ####################################################


    #pose list npy로 저장
    #print(f"pose_list[0]={pose_list[0]}")
    #print(f"pose_list[0].shape={pose_list[0].shape}")

    '''
    for i in range(len(pose_list) - 1):
        current_pose = pose_list[i]
        next_pose = pose_list[i + 1]
        
        # 다른 interpolation method 찾아보자! 일단 naive하게 평균..
        middle_pose = (current_pose + next_pose) / 2.0
        
        interpolated_pose_list.append(current_pose)
        interpolated_pose_list.append(middle_pose)

    interpolated_pose_list.append(pose_list[-1])
    '''
    '''
    pose_list_reshaped = pose_list.reshape(pose_list.shape[0], -1)

    original_indices = np.arange(pose_list.shape[0])
    new_indices = np.linspace(0, pose_list.shape[0] - 1, pose_list.shape[0] * 5 - 1) 

    interpolated_poses = []
    for i in range(pose_list_reshaped.shape[1]):
        interp_func = interp1d(original_indices, pose_list_reshaped[:, i], kind='cubic')
        interpolated_poses.append(interp_func(new_indices))

    new_pose_list = np.array(interpolated_poses).T.reshape(-1, 3, 3)
    print(f'new_pose_list.shape:{new_pose_list.shape}')
    '''
    '''
    print(f'pose_list.shape:{pose_list.shape}')

    quaternions = [R.from_matrix(pose).as_quat() for pose in pose_list]
    print(f'quaternions[0].shape:{quaternions[0].shape}')
    print(f'len(quaternions):{len(quaternions)}')

    original_indices = np.linspace(0, 1, len(quaternions))
    target_indices = np.linspace(0, 1, len(quaternions) * 4 - 1)

    # SLERP
    rotations = [R.from_quat(quat) for quat in quaternions]
    slerped_rotations = R.slerp(rotations, original_indices, target_indices)

    new_pose_list = slerped_rotations.as_matrix()
    print(f'new_pose_list.shape:{new_pose_list.shape}')
    '''
    '''
    quaternions = [R.from_matrix(pose).as_quat() for pose in pose_list] # -> (50,35,4)
    print(f'len(quaternions):{len(quaternions)}')
    print(f'quaternions[0].shape:{quaternions[0].shape}')

    #rotations = [R.from_quat(quat) for quat in quaternions]
    rotations = R.from_quat(quaternions)
    print(f'roatations:{rotations}')
    print(f'len(rotations):{len(rotations)}')
    #print(f'rotations[0].shape:{rotations[0].shape}')
    
    original_indices = np.linspace(0, 1, len(quaternions))
    target_indices = np.linspace(0, 1, len(quaternions) * 4 - 1)

    slerp = Slerp(original_indices, rotations)
    slerped_rotations = slerp(target_indices)

    new_pose_list = slerped_rotations.as_matrix()
    print(f'new_pose_list.shape:{new_pose_list.shape}')
    '''
    
    print(f'n_pose_list.shape:{n_pose_list.shape}')
    quaternions = np.array([R.from_matrix(pose).as_quat() for pose in n_pose_list])  # -> (50, 35, 4)

    original_indices = np.linspace(0, 1, len(n_pose_list))
    target_indices = np.linspace(0, 1, len(n_pose_list) * 3 - 3)

    new_pose_list = np.zeros((len(target_indices), 35, 3, 3), dtype=np.float32)

    for joint in range(35):
        joint_quaternions = quaternions[:, joint, :]

        rotations = R.from_quat(joint_quaternions)
        slerp = Slerp(original_indices, rotations)
        
        slerped_rotations = slerp(target_indices)
        
        new_pose_list[:, joint, :, :] = slerped_rotations.as_matrix()

    print(f'new_pose_list.shape: {new_pose_list.shape}')
    
    #save_path = '/root/dev/oneshotgart/GART/walking_pose_list.npy'
    #np.save(save_path, new_pose_list)
    #################################
    quaternions2 = np.array([R.from_matrix(pose2).as_quat() for pose2 in n_pose_list2])  # -> (50, 35, 4)

    original_indices2 = np.linspace(0, 1, len(n_pose_list2))
    target_indices2 = np.linspace(0, 1, len(n_pose_list2) * 3 - 3)

    new_pose_list2 = np.zeros((len(target_indices2), 35, 3, 3), dtype=np.float32)

    for joint2 in range(35):
        joint_quaternions2 = quaternions2[:, joint2, :]

        rotations2 = R.from_quat(joint_quaternions2)
        slerp2 = Slerp(original_indices2, rotations2)
        
        slerped_rotations2 = slerp2(target_indices2)
        
        new_pose_list2[:, joint2, :, :] = slerped_rotations2.as_matrix()
    ##########################################
    #3
    #################################
    '''
    quaternions3 = np.array([R.from_matrix(pose3).as_quat() for pose3 in n_final_pose_list])  # -> (50, 35, 4)

    original_indices3 = np.linspace(0, 1, len(n_final_pose_list))
    target_indices3 = np.linspace(0, 1, len(n_final_pose_list) * 3 - 3)

    new_final_pose_list = np.zeros((len(target_indices3), 35, 3, 3), dtype=np.float32)

    for joint3 in range(35):
        joint_quaternions3 = quaternions3[:, joint3, :]

        rotations3 = R.from_quat(joint_quaternions3)
        slerp3 = Slerp(original_indices3, rotations3)
        
        slerped_rotations3 = slerp2(target_indices3)
        
        new_final_pose_list[:, joint3, :, :] = slerped_rotations3.as_matrix()
        '''
    ##########################################
    ##########################################
    #dance
    quaternions_dance = np.array([R.from_matrix(pose_dance).as_quat() for pose_dance in new_pose_list_dance])  # -> (50, 35, 4)

    original_indices_dance = np.linspace(0, 1, len(new_pose_list_dance))
    target_indices_dance = np.linspace(0, 1, len(new_pose_list_dance) * 3 - 3)

    new_pose_list_dancing = np.zeros((len(target_indices_dance), 35, 3, 3), dtype=np.float32)

    for joint_dance in range(35):
        joint_quaternions_dance = quaternions_dance[:, joint_dance, :]

        rotations_dance = R.from_quat(joint_quaternions_dance)
        slerp_dance = Slerp(original_indices_dance, rotations_dance)
        
        slerped_rotations_dance = slerp_dance(target_indices_dance)
        
        new_pose_list_dancing[:, joint_dance, :, :] = slerped_rotations_dance.as_matrix()


    ##########################################



    animation = matrix_to_axis_angle(torch.from_numpy(new_pose_list)).to(solver.device) # original walk
    print(f'animation.shape: {animation.shape}')
    animation2 = matrix_to_axis_angle(torch.from_numpy(new_pose_list2)).to(solver.device) # drone

    animation3 = matrix_to_axis_angle(torch.from_numpy(pose_list_gart)).to(solver.device) #gart_original

    #animation3 = torch.from_numpy(new_pose_smal_list).to(solver.device) # robot
    #animation4 = matrix_to_axis_angle(torch.from_numpy(n_final_pose_list)).to(solver.device)
    animation4 = torch.from_numpy(n_final_pose_list).to(solver.device) # refine robot
    animation5 = matrix_to_axis_angle(torch.from_numpy(new_pose_list_dancing)).to(solver.device) # dance


    pose_front_tensor = matrix_to_axis_angle(torch.from_numpy(pose_front_array)).to(solver.device)
    pose_front_tensor = torch.squeeze(pose_front_tensor, 1)
    #print(animation.dtype, pose.dtype)
    #animation.dtype = pose.dtype
    animation[:, [32, 33, 34]] = pose[:, [32, 33, 34]]
    animation2[:, [32, 33, 34]] = pose[:, [32, 33, 34]]
    animation4[:, [32, 33, 34]] = pose[:, [32, 33, 34]]
    animation3[:, [32, 33, 34]] = pose[:, [32, 33, 34]]
    animation5[:, [32, 33, 34]] = pose[:, [32, 33, 34]]
    print(f'animation[:,[0]].shape= {animation[:,[0]].shape}') # (324, 1, 3)
    #print(f'pose_front_tensor.shape= {pose_front_tensor.shape}')
    print(f'pose_front_tensor[:, 0].shape= {pose_front_tensor[:, 0].shape}') #(1,1,3)
    #animation[:,[0]] = pose_front_tensor[:, 0] # #for frontal walk
    #animation2[:, [32, 33, 34]] = pose[:, [32, 33, 34]]
    #animation[:,[0]] = pose_front_tensor[:, 0]
    #animation3[:,[0]] = pose_front_tensor[:, 0]
    animation4[:,[0]] = pose_front_tensor[:, 0]
    #animation5[:,[0]] = pose_front_tensor[:, 0]
    print(f'animation4.shape= {animation4.shape}')

    animations = [animation2, animation3, animation4]  # Define animations
    #animation_names = ['animation_walk', 'animation2_drone', 'animation3_robot', 'animation4_refine_robot', 'animation5_dance']
    animation_names = ['animation2_drone','animation3_gart_original', 'animation4_front_view']

    #animations = [animation4]  # Define animations
    #animation_names = ['animation4_refine_robot']
    #limbs = [limb, limb, limb, limb, limb]  # Assuming same limb for simplicity, adjust as needed
    
    #save_path1=osp.join(viz_dir, f"animation_{seq_name}_orginal_walk.gif")
    save_path2=osp.join(viz_dir, f"animation_{seq_name}_drone.gif")
    save_path3=osp.join(viz_dir, f"animation_{seq_name}_gart_original.gif")
    save_path4=osp.join(viz_dir, f"animation_{seq_name}_front_view.gif")

    save_path_animation_img = osp.join(viz_dir, f"{seq_name}_animation_img")
    
    #save_path5=osp.join(viz_dir, f"animation_{seq_name}_dance.gif")

    #save_paths = [save_path1, save_path2, save_path3, save_path4, save_path5]  # Define your save paths
    save_paths = [save_path2, save_path3, save_path4]
    #save_paths = [save_path5]  # Define your save paths


    viz_dog_animation(
        model.to("cuda"),
        animations,
        animation_names,
        save_path_animation_img,
        limb,
        trans,
        data_provider.H,
        data_provider.W,
        data_provider.K_list[0],
        save_paths,
        seq_name,
        #save_path3=osp.join(viz_dir, f"animation_{seq_name}_walk_front.gif"),
        fps=90,
        device="cuda:0"
    )
    return
