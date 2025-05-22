gpus=4
cofig_file=configs/semantic/eval_large_reason.yaml

#pc59
python train_net.py --config-file $cofig_file --num-gpus $gpus --eval-only \
 MODEL.TEST_CLASS_REASON_JSON "reason_data/general_reason/pc59_class_des.json" \
 MODEL.REASON_DIR "reason_data/image_reason/Qwen2.5-VL-72B-Instruct-AWQ"\
 MODEL.REASON_DATA "pc59"\
 DATASETS.TEST "('openvocab_pascal_ctx59_sem_seg_val',)" \
 OUTPUT_DIR ./out/semantic/evaluation_large_reason/pc59/