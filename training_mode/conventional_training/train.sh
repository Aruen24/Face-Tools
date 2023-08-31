mkdir 'log'
python train.py \
#   # --data_root '/home/hutian/faces_tx_tx2_ms_asia_vgg_id_stu_298424_160_144_80/' \
    --data_root './no_kd_data_298424/' \
	  --train_file '' \
    --backbone_type 'vargfacenet' \
    --backbone_conf_file '../backbone_conf.yaml' \
    --head_type 'ArcFace' \
    --head_conf_file '../head_conf.yaml' \
    --lr 0.1 \
    --out_dir '/home/mapengfei/data/FR/vargfacenet/save/mask_r100_arcface_no_kd_298424' \
    --epoches 37 \
    --step '14, 25, 33' \
    --print_freq 200 \
    --save_freq 3000 \
    --batch_size 800 \
    --momentum 0.9 \
    --log_dir 'log' \
    --tensorboardx_logdir 'new_arcface_172038_kd' \
    --pretrain_model '/home/mapengfei/data/FR/vargfacenet/save/mask_r100_arcface_no_kd_298424/Epoch_model_head_20.pt' \
    -r \
    2>&1 | tee log/log.log
