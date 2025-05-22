gpus=4
config_file=configs/semantic/eval_base_reason.yaml


#  #pc59
python train_net.py --config-file $config_file --num-gpus $gpus --eval-only \
 MODEL.TEST_CLASS_REASON_JSON "reason_data/general_reason/pc59_class_des.json" \
 MODEL.REASON_DIR "reason_data/reason_data/image_reason/Qwen2.5-VL-72B-Instruct-AWQ"\
 MODEL.REASON_DATA "pc59"\
 MODEL.CLASS_NAME_JSON "datasets/pc59.json" \
 DATASETS.TEST "('openvocab_pascal_ctx59_sem_seg_val',)" \
 OUTPUT_DIR ./out/semantic/evaluation_base_reason/pc59/
