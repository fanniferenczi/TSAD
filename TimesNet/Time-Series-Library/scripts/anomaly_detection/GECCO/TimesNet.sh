export CUDA_VISIBLE_DEVICES=0

# Step 1: Download and prepare the GECCO dataset
python -u scripts/prepare_gecco.py

# Step 2: Run TimesNet anomaly detection
# GECCO 2018 has 9 sensor features (Tp, Cl, pH, Redox, Leit, Trueb, Trueb_FNU, Fm, + possible extras)
# The actual number of features will be printed by prepare_gecco.py
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/GECCO \
  --model_id GECCO \
  --model TimesNet \
  --data GECCO \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 32 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 9 \
  --c_out 9 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3
