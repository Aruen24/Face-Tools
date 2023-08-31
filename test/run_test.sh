#sleep 60000

db_name=$1
model_name=$2

python store_feature.py --db $db_name --model $model_name --anchor_pics_path ./test_data/no_mask/anchor

python test_validate_1vN.py --db $db_name --model $model_name --val_pics_path ./test_data/no_mask/not_in
python test_validate_1vN.py --db $db_name --model $model_name --val_pics_path ./test_data/no_mask/in


#python store_feature.py --db $db_name --model $model_name --anchor_pics_path /home/hutian/src/insightface/src/test/datasets/pass_histroy_test3/test_anchor/8674

#python test_validate_1vN.py --db $db_name --model $model_name --val_pics_path /home/hutian/src/insightface/src/test/datasets/pass_histroy_test3/test_data/no_mask/not_in
#python test_validate_1vN.py --db $db_name --model $model_name --val_pics_path /home/hutian/src/insightface/src/test/datasets/pass_histroy_test3/test_data/no_mask/in


#python att_store_feature.py --db $db_name --model $model_name --anchor_pics_path datasets/pass_histroy_test_160/test_anchor/8674

#python att_test_validate_1vN.py --db $db_name --model $model_name --val_pics_path datasets/pass_histroy_test_160/test_data/mask/in
#python att_test_validate_1vN.py --db $db_name --model $model_name --val_pics_path datasets/pass_histroy_test_160/test_data/mask/not_in


#python att_store_feature.py --db $db_name --model $model_name --anchor_pics_path /home/ubuntu/qqqq/vip_out
#python att_test_validate_1vN.py --db $db_name --model $model_name --val_pics_path /home/ubuntu/qqqq/18_floor_img_144

