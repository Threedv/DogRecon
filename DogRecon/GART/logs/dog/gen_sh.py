import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', required=True, type=str, help='the path to the source video')

    opt = parser.parse_args()
    image_name = opt.image_name #.split('/')[-1].split('.')[0]

    commands = []
    commands.append('#!/bin/bash')
    #steps = 5
    #commands.append(f'echo ========================================')
    #commands.append(f'echo 1/{steps}: Stable Zero123')
    #commands.append(f'echo ========================================')

    for i in range(1, 67):  # 66개의 명령어 생성
        #command = f"python preprocess/gen_run_flip.py --image_name ./oneshot_image/d{i}_flip.png"
        command=f'cp seq\=dog_{i}_flip_prof\=dog_data\=dog_demo/_dog_viz/animation.gif ./ori/animation{i}.gif'
        command1=f'cp seq\=dog_{i}_flip_prof\=dog_data\=dog_demo/_dog_viz/spin.gif ./ori/spin1_{i}.gif'
        command2=f'cp seq\=dog_{i}_flip_prof\=dog_data\=dog_demo/_dog_viz/spin2.gif ./ori/spin2_{i}.gif'

        commands.append(command)
        commands.append(command1)
        commands.append(command2)









    print(*commands, sep='\n')
    with open(f"run_shsh_{image_name}.sh", "w") as outfile:
        outfile.write("\n".join(commands))

if __name__ == "__main__":
    main()
