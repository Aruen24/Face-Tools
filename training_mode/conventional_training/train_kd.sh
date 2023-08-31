mkdir 'log'
python train_kd.py \
    #--data_root '/home/mapengfei/Face-Recognition/mask/FaceX-Zoo-no-kd_298424/training_mode/conventional_training/kd_data_298424/' \
    --data_root './kd_data_298424/' \
    --train_file '' \
    --backbone_type 'vargfacenet' \
    --backbone_conf_file '../backbone_conf.yaml' \
    --head_type 'ArcFace' \
    --head_conf_file '../head_conf.yaml' \
    --lr 0.1 \
    --out_dir '/home/mapengfei/data/FR/vargfacenet/save/mask_vargfacenet_arcface_kd_298424' \
    --epoches 40 \
    --step '13, 27, 36' \
    --print_freq 200 \
    --save_freq 3000 \
    --batch_size 800 \
    --momentum 0.9 \
    --log_dir 'log' \
    --tensorboardx_logdir 'new_arcface_172038_kd' \
    --pretrain_model '/home/mapengfei/data/FR/vargfacenet/save/mask_vargfacenet_arcface_kd_298424/Epoch_model_head_39.pt' \
    -r \
    2>&1 | tee log/0506.log