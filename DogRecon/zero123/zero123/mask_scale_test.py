import numpy as np
import cv2
import sys

def expand_white_area(mask, expansion_size):
    # 흰색 부분 찾기
    white_area = mask == 1
    
    kernel = np.ones((expansion_size, expansion_size), np.uint8)
    
    # expand 시작
    expanded_mask = cv2.dilate(white_area.astype(np.uint8), kernel, iterations=1)
    
    # 원래 mask 크기로 유지
    expanded_mask = expanded_mask * 1  # 흰색 부분 유지
    return expanded_mask

if __name__ == "__main__":
    # Get arguments from the command line
    mask_path = sys.argv[1]
    lv2_output_path = sys.argv[2]
    lv3_output_path = sys.argv[3]
    back_output_path = sys.argv[4]

    # Load the original mask
    mask = np.load(mask_path)

    # Generate lv2 and lv3 masks
    lv2_mask = expand_white_area(mask, 3) - mask
    lv100_mask = expand_white_area(mask, 40) - expand_white_area(mask, 3)
    back_mask = 1 - expand_white_area(mask, 40)

    # Save the masks as .npy files
    np.save(lv2_output_path, lv2_mask)
    np.save(lv3_output_path, lv100_mask)
    np.save(back_output_path, back_mask)
    
    



