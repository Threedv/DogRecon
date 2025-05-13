# take the instant avatar format
import numpy as np
import os.path as osp
import numpy as np
import glob
import imageio
from tqdm import tqdm
from transforms3d.euler import euler2mat
from pycocotools import mask as masktool

import torch
from torch.utils.data import Dataset
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
#이게 가장 중요할 수 도~ 오 range 를 정해놨네 .의미있는 smal값이 예측된 곳들.

TRAIN_RANGES = {
    "hound_": [[178, 220], [262, 310], [339, 441]],
    "hound": [[22, 60], [130, 144], [150, 223], [250, 441]],
    "french": [[150, 384], [468, 530]],
    "german": [[2, 45], [93, 284], [291, 400]],
    "alaskan": [[65, 90], [153, 185], [255, 325], [350, 440], [630, 667]],
    "pit_bull": [[100, 127], [129, 492], [629, 808]],
    "irish": [
        [90, 127],
        [328, 384],
        [390, 400],
        [430, 470],
        [549, 564],
        [584, 625],
        [881, 1008],
    ],
    "english": [[164, 264], [292, 615], [660, 664]],
    "shiba": [[75, 200], [317, 343], [366, 500]],
    "corgi": [
        [30, 66],
        [150, 190],
        [253, 266],
        [495, 513],
        [573, 588],
        [639, 660],
        [779, 796],
        [829, 860],
    ],
    "oneshot_french":[[0,6]],
     "hound_": [[178, 220], [262, 310], [339, 441]],
    "hound": [[22, 60], [130, 144], [150, 223], [250, 441]],
    "hound_fewshot": [[44,44], [130, 130], [566,566]],
    "french": [[150, 384], [468, 530]],
    "french_fewshot": [[150, 150], [468, 468],[799,799]],
    "german": [[2, 45], [93, 284], [291, 400]],
    "german_fewshot": [[2, 2], [93, 93], [538, 538]],
    "alaskan": [[65, 90], [153, 185], [255, 325], [350, 440], [630, 667]],
    "alaskan_fewshot": [[65, 65], [255, 255], [495,495]],
    "pit_bull": [[100, 127], [129, 492], [629, 808]],
    "pit_bull_fewshot": [[100, 100], [495, 495], [629, 629]],
    "irish": [
        [90, 127],
        [328, 384],
        [390, 400],
        [430, 470],
        [549, 564],
        [584, 625],
        [881, 1008],
    ],
    "english": [[164, 264], [292, 615], [660, 664]],
    "english_fewshot": [[164, 164], [292, 292], [853,853]],
    "shiba": [[75, 200], [317, 343], [366, 500]],
    "shiba_fewshot": [[144, 144], [317, 317], [366, 366]],
    "corgi": [
        [30, 66],
        [150, 190],
        [253, 266],
        [495, 513],
        [573, 588],
        [639, 660],
        [779, 796],
        [829, 860],
    ],
    "corgi_fewshot": [
        [30, 30],
        [253, 253],
        [853, 853],
    ],
    "wolf" :[[10,172]],
    "wolf_fewshot" :[[10,10],[50,50],[172,172]],
    "wolf_resize" :[[10,172]],
    "beagle_dog" : [[100,362]],
    "beagle_dog_fewshot":[[100,100],[121,121],[361,361]],
    "beagle_dog_resize":[[100,362]],
    "french_dogrecon": [[0,144]],
    "german_dogrecon": [[0,144]],
    "official_alaskan_571_flip": [[0,161]], # 수정.. 160에 143이 들어감.. 왜? 허허. 일단 ok
    "pit_bull_dogrecon": [[0,144]],
    "hound_dogrecon": [[0,144]],
    "english_dogrecon": [[0,144]],
    "shiba_dogrecon":[[0,144]],
    "corgi_dogrecon": [[0,144]],
    "wolf_dogrecon":[[0,144]],
    "beagle_dog_dogrecon":[[0,144]],
    "official_corgi_0853_flip":[[0,143]],
    "wolf_damage_172_0024_flip":[[0,143]],#[[0,153]],
    'beagle_121_0002_flip':[[0,143]],
    "official_alaskan_571_flip_ijcv": [[0,161]],
    "official_corgi_0853_flip_ijcv":[[0,143]],
    "wolf_damage_172_0024_flip_ijcv":[[0,143]],
    'beagle_121_0002_flip_ijcv':[[0,143]],
}

TEST_RANGES = {
    #"french": [[635, 815]],
    #"german": [[421, 550]],
    #"alaskan": [[493, 565]],
    #"pit_bull": [[493, 628]],
    #"hound": [[442, 571]],
    #"english": [[678, 883]],
   # "irish": [[1136, 1274]],
    #"shiba": [[501, 650]],
    #"corgi": [[861, 877]],
    "oneshot_french":[[0,6]],
    "french": [[635, 799],[800,815]],
    "german": [[421, 538],[539,550]],
    "alaskan": [90,152,186,197,202,216,334], #usrud#[[496,565]],
    "pit_bull": [[496,628]],
    "hound": [[442, 566],[567,571]],
    "english": [[678, 853],[854,883]],
    "irish": [[1136, 1274]],
    "shiba": [[501, 650]],
    "corgi": [861,863,865,867,869,871,873] ,#[[861, 877]],
    "wolf" : [[0,10]],
    "beagle_dog":[[0,100]],
    "wolf_resize" : [102,104,106,108,110,112],#[0,2,4,6,8,10,12],#[[0,10]],
    "beagle_dog_resize":[0,60,68,76,107,254,363],#[[0,100]],
    "french_fewshot": [[635, 799],[800,815]],
    "german_fewshot": [[421, 538],[539,550]],
    "alaskan_fewshot": [90,152,186,197,202,216,334], #변경  org [[493, 495],[496,565]]
    "pit_bull_fewshot": [[496,628]], #변경  org [[493, 495],[496,565]]
    "hound_fewshot": [[442, 566],[567,571]],
    "english_fewshot": [[678, 853],[854,883]],
    "shiba_fewshot": [[501, 650]],
    "corgi_fewshot": [861,863,865,867,869,871,873],
    "wolf_fewshot":[102,104,106,108,110,112],#
    "beagle_dog_fewshot":[0,60,68,76,107,254,363],#[[0,100]],
    "french_dogrecon": [[635, 799],[800,815]],
    "german_dogrecon": [[421, 538],[539,550]],
    "official_alaskan_571_flip": [90,152,186,197,202,216,334],
    "official_alaskan_571_flip_ijcv": [90,152,186,197,202,216,334],
    "official_corgi_0853_flip": [861,863,865,867,869,871,873],
    "pit_bull_dogrecon": [[493, 495],[496,628]],
    "hound_dogrecon": [[442, 566],[567,571]],
    "english_dogrecon": [[678, 853],[854,883]],
    "shiba_dogrecon": [[501, 650]],
    "corgi_dogrecon": [[861, 877]],
    "wolf_dogrecon":[[0,10]],
    "beagle_dog_dogrecon":[[0,100]],
    "wolf_damage_172_0024_flip":[102,104,106,108,110,112],#[0,2,4,8,10,12],
    "beagle_121_0002_flip":[0,60,68,76,107,254,363],
}

# alaskan alaskan_fewshot beagle_dog beagle_dog_fewshot corgi corgi_fewshot english english_fewshot french french_fewshot german german_fewshot hound hound_fewshot pit_bull pit_bull_fewshot shiba shiba_fewshot wolf wolf_fewshot
'''
def get_frame_id_list(video_name):
    if video_name in ["hound_","hound", "french", "german", "alaskan", "pit_bull", "irish","english","shiba","corgi"]:
    #if video_name in ["hound_","hound", "french", "german", "alaskan", "pit_bull", "irish","english","shiba","corgi", "pit_bull_fewshot"]:
        bounds = TRAIN_RANGES[video_name]
    elif video_name.split('_')[-1] == 'gart':
        bounds = TRAIN_RANGES[video_name]
        #bounds=[[0,10], [55,71]]
    elif video_name.split('_')[0] =='blend':
        bounds=[[0,18],[54,71]]
    elif video_name.split('_')[-1] == 'flip':
        bounds = [[0,143]]
    elif video_name.split('_')[-1] == 'artemis':
        bounds=[[0,10], [55,71]]
    elif video_name.split('_')[-1] == 'fewshot':
        bounds = TRAIN_RANGES[video_name]
    else:
        bounds = [[0,71]]
    ids = []
    for b in bounds:
        ids += list(range(b[0], b[1] + 1))
    return ids
'''

def get_frame_id_list(video_name):
    if video_name in ['beagle_121_0002_flip','wolf_damage_172_0024_flip','official_alaskan_571_flip', 'alaskan', 'alaskan_fewshot', 'official_corgi_0853_flip','alaskan alaskan_fewshot', 'beagle_dog_resize', 'beagle_dog_fewshot', 'corgi', 'corgi_fewshot', 'english', 'english_fewshot', 'french', 'french_fewshot', 'german', 'german_fewshot', 'hound', 'hound_fewshot', 'pit_bull', 'pit_bull_fewshot', 'shiba', 'shiba_fewshot', 'wolf_resize', 'wolf_fewshot']:
    #if video_name in ["hound_","hound", "french", "german", "alaskan", "pit_bull", "irish","english","shiba","corgi", "pit_bull_fewshot"]:
        bounds = TRAIN_RANGES[video_name]
    else:
        bounds = [[0,71]]
        #bounds = [[0,143]]
    ids = []
    for b in bounds:
        ids += list(range(b[0], b[1] + 1))
    return ids


def get_test_frame_id_list(video_name, test_size=15):
    bounds = TEST_RANGES[video_name]
    # ids = []
    # for b in bounds:
    #     ids += list(range(b[0], b[1] + 1, (b[1] - b[0] + 1) // test_size))
    # ids = ids[:test_size]
    # 수정 
    return bounds


class Dataset(Dataset):
    def __init__(
        self, data_root="data/dog_data", video_name="hound", test=False
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.video_name = video_name
        root = osp.join(data_root, video_name)

        image_dir = osp.join(root, "images")
        pose_dir = osp.join(root, "pred")

        if test:
            id_list = get_test_frame_id_list(video_name)
        else:
            id_list = get_frame_id_list(video_name)

        self.rgb_list, self.mask_list = [], []
        betas_list = []
        self.pose_list, self.trans_list = [], []
        self.K_list = []

        for i in tqdm(id_list):
            img_path = osp.join(image_dir, f"{i:04d}.png")
            msk_path = osp.join(image_dir, f"{i:04d}.npy")
            pose_path = osp.join(pose_dir, f"{i:04d}.npz")
            if not osp.exists(msk_path):
                continue

            rgb = imageio.imread(img_path)
            assert rgb.shape[0] == 512 and rgb.shape[1] == 512
            #if video_name in ["hound_","hound", "french", "german", "alaskan", "pit_bull", "irish","english","shiba","corgi", "pit_bull_fewshot" ]:
            try:
                #video_name in ['official_corgi_0853_flip','official_alaskan_571_flip', 'alaskan', 'alaskan_fewshot', 'alaskan_fewshot','corgi', 'corgi_fewshot', 'english', 'english_fewshot', 'french', 'french_fewshot', 'german', 'german_fewshot', 'hound', 'hound_fewshot', 'pit_bull', 'pit_bull_fewshot', 'shiba', 'shiba_fewshot']:
                
                mask = np.load(msk_path, allow_pickle=True).item()
                mask = masktool.decode(mask)

            except:
                mask = np.load(msk_path, allow_pickle=True)
            

            pred = dict(np.load(pose_path, allow_pickle=True))
            betas = pred["pred_betas"]
            betas_limbs = pred["pred_betas_limbs"]

            pose = pred["pred_pose"]
            pose = matrix_to_axis_angle(torch.from_numpy(pose)).numpy()[0].reshape(-1)
            trans = pred["pred_trans"][0]
            focal = pred["pred_focal"] * 2  # for 512 size image

            K = np.eye(3)
            K[0, 0], K[1, 1] = focal, focal
            K[0, 2], K[1, 2] = 256, 256

            rgb = (rgb[..., :3] / 255).astype(np.float32)
            #print(f'mask:{mask}')
            mask = mask.astype(np.float32)
            # apply mask
            rgb = rgb * mask[..., None] + (1 - mask[..., None])

            self.rgb_list.append(rgb)
            self.mask_list.append(mask)
            betas_list.append(betas)
            self.pose_list.append(np.concatenate([pose, betas_limbs[0]], 0))
            self.trans_list.append(trans)
            self.K_list.append(K)
        # average the beta
        self.betas = np.concatenate(betas_list, 0).mean(0)
        print(f"Loaded {len(self.rgb_list)} frames from {video_name}")
        return

    def __len__(self):
        return len(self.rgb_list)

    def __getitem__(self, idx):
        img = self.rgb_list[idx]
        msk = self.mask_list[idx]
        pose = self.pose_list[idx]

        ret = {
            "rgb": img.astype(np.float32),
            "mask": msk,
            "K": self.K_list[idx].copy(),
            "smpl_beta": self.betas,
            "smpl_pose": pose,
            "smpl_trans": self.trans_list[idx],
            "idx": idx,
        }

        meta_info = {
            "video": self.video_name,
        }
        viz_id = f"video{self.video_name}_dataidx{idx}"
        meta_info["viz_id"] = viz_id
        return ret, meta_info


if __name__ == "__main__":
    dataset = Dataset(data_root="../data/dog_demo")
    ret = dataset[0]
