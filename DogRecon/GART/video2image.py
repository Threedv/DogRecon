import cv2
import os

# 비디오 파일 경로
video_path = './data/dog_data_official/animation_test_drone/images/animation_test_drone.mov'
# 프레임을 저장할 디렉토리 경로
output_dir = './data/dog_data_official/animation_test_drone/images'

# 출력 디렉토리가 없으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 비디오 파일 열기
cap = cv2.VideoCapture(video_path)

# 비디오의 총 프레임 수와 FPS(초당 프레임 수)를 가져옵니다.
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'total_frames={total_frames}')
fps = cap.get(cv2.CAP_PROP_FPS)
print(f'fps={fps}')

# 추출할 프레임의 간격을 설정합니다. 예: 1초마다 한 프레임
frame_interval = int(fps)  # 1초마다

# 프레임을 순회하면서 이미지로 저장
for frame_number in range(0, total_frames, 1):
    # 프레임 번호를 설정하여 해당 프레임으로 이동
    if fps>30:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number*2)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # 현재 프레임을 읽기
    success, frame = cap.read()
    
    # 성공적으로 읽었다면 이미지로 저장
    if success:
        # 저장할 이미지 파일 경로
        output_path = os.path.join(output_dir, f'{frame_number:04d}.png')
        # 이미지 파일로 저장
        cv2.imwrite(output_path, frame)
    else:
        break

# 비디오 파일 닫기
cap.release()
print('프레임 추출 완료.')