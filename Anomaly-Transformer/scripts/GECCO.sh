export CUDA_VISIBLE_DEVICES=2,3

python main.py --anormly_ratio 0.05 --num_epochs 10  --batch_size 128  --mode train --dataset GECCO  --data_path dataset/GECCO --input_c 9  --output_c 9
python main.py --anormly_ratio 0.05 --num_epochs 10  --batch_size 128  --mode test  --dataset GECCO  --data_path dataset/GECCO --input_c 9  --output_c 9  --pretrained_model 10
