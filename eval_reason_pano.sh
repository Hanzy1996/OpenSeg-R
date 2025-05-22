gpus=4
cofig_file=configs/panoptic/eval_pano_reason.yaml


#ade150
python train_net.py --config-file $cofig_file --num-gpus $gpus --eval-only \
 MODEL.TEST_CLASS_REASON_JSON "reason_data/general_reason/ade150_class_des.json" \
 MODEL.REASON_DIR "reason_data/image_reason/Qwen2.5-VL-72B-Instruct-AWQ"\
 MODEL.REASON_DATA 'ade150'\
 MODEL.CLASS_NAME_JSON "datasets/ade150.json" \
 DATASETS.TEST "('openvocab_ade20k_panoptic_val',)" \
 OUTPUT_DIR ./out/panoptic/evaluation_large_reason/ade150/
