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

    for i in range(2,70):  # 66개의 명령어 생성
        #command = f"python preprocess/gen_run_flip.py --image_name ./oneshot_image/d{i}_flip.png"
        command0="conda activate bite"
        command1=f'python scripts/full_eccv.py --workers 12 --config refinement_cfg_test_withvertexwisegc_csaddnonflat_crops.yaml --model-file-complete cvpr23_dm39dnnv3barcv2b_refwithgcpervertisflat0morestanding0/checkpoint.pth.tar --suffix ttopt_vtest1 --image_path d{i}_eccv_flip'
        command2=f'python geometricprior_supple.py --image_name d{i}_eccv_flip'
        command3=f'python geometricprior_supple_flip.py --image_name d{i}_eccv_flip'
        command4='conda deactivate'
        command5=f'cd ../GART/'
        command6=f'conda activate gart'
        command7=f'python solver.py --profile ./profiles/dog/dog.yaml --dataset dog_demo --seq d{i}_eccv_flip --logbase dog --no_eval --semantic clip --sampling mask'
        command8=f'conda deactivate'
        command9=f'cd ../bite_release/'


        


        commands.append(command0)
        commands.append(command1)
        commands.append(command2)
        commands.append(command3)
        commands.append(command4)
        commands.append(command5)
        commands.append(command6)
        commands.append(command7)
        commands.append(command8)
        commands.append(command9)

        









    print(*commands, sep='\n')
    with open(f"run_shsh_{image_name}.sh", "w") as outfile:
        outfile.write("\n".join(commands))

if __name__ == "__main__":
    main()