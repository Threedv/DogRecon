import os
import numpy as np
from scipy.interpolate import interp1d

# 보간 작업을 적용할 디렉토리
input_dir = '/home/user/gs/huggstudy/GART/novel_poses/ijcv_dog_motion/ijcv_pred'
output_dir = '/home/user/gs/huggstudy/GART/animals_smal_joints_interpolation'

# 입력 디렉토리의 모든 npy 파일에 대해 반복
for filename in os.listdir(input_dir):
    if filename.endswith('.npy'):
        # 파일 로드
        file_path = os.path.join(input_dir, filename)
        data = np.load(file_path)

        # 기존의 frame 수 N
        N = data.shape[0]

        # 새로운 frame 수 (5배 증가)
        new_N = 5 * N

        # 원래의 인덱스와 새로운 인덱스 생성
        original_indices = np.arange(N)
        new_indices = np.linspace(0, N - 1, new_N)

        # 보간 함수 생성 및 적용
        interp_func = interp1d(original_indices, data, axis=0, kind='linear')
        new_data = interp_func(new_indices)

        # 새로운 파일 이름 생성 및 저장
        output_filename = f"{os.path.splitext(filename)[0]}_interpolated.npy"
        output_file_path = os.path.join(output_dir, output_filename)
        np.save(output_file_path, new_data)

        print(f"Processed and saved: {output_file_path}")

print("All files have been processed and saved.")
