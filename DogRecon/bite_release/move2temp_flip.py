import numpy as np
import argparse
import os
import shutil



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', required=True, type=str, help='the path to the source video')
    opt = parser.parse_args()
    image_name = opt.image_name
    mv1 = np.load(f'../GART/data/dog_data_official/{image_name}/pred/move2.npy')
    source_folder = f'../GART/data/dog_data_official/{image_name}/images/'
    destination_folder = f'../GART/data/dog_data_official/{image_name}/images_temp/'
    for i in range(72,144):
    # Path of the file to be moved
        if i< mv1[0] or i > mv1[1]:
            filename = f'{i:04d}.png'
            file_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)
            # Move the file to the destination folder
            print(file_path,destination_path)
            shutil.move(file_path, destination_path)

if __name__ == "__main__":
    main()