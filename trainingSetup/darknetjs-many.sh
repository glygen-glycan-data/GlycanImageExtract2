#!/bin/sh
set -euo pipefail

# Input file format...
#
#   gd:YOLO/YOLOLinkNoLink-Uniform-10k-2025-10-28 linknolink_v5.1 --iou 0.85 --conf 0.9 --nms_kind greedynms --beta_nms 0.85 --max_batches 20000 
#   gd:YOLO/YOLOLinkNoLink-Uniform-10k-2025-10-28 linknolink_v5.2                       --nms_kind greedynms --beta_nms 0.85 --max_batches 20000 
#   gd:YOLO/YOLOLinkNoLink-Uniform-10k-2025-10-28 linknolink_v5.3 --iou 0.85 --conf 0.9                                      --max_batches 20000 
#   gd:YOLO/YOLOLinkNoLink-Uniform-10k-2025-10-28 linknolink_v5.4                                                            --max_batches 20000

# For fine-tuninig provide --weights and --config should be the same the yolo darknet model used to build the base pre-trained model
#   gd:YOLO/monos_finetune monos_v9_1_finetune --config yolov4-tiny --weights gd:YOLO/YOLO_58/mono_v5_8.weights --iou 0.85 --conf 0.9 --max_batches 20000 
#   gd:YOLO/monos_finetune monos_v9_2_finetune --config yolov4-tiny --weights https://drive.google.com/uc?id=18l43XAtscph_Ywjgh6yXwIWn7geO3a5W --iou 0.85 --conf 0.9 --learning_rate 0.0005 --max_batches 20000 


while IFS= read -r line; do
    arg=($line)
    imagedir="${arg[0]}"
    jobname="${arg[1]}" 
    arg=("${arg[@]:2}")
    if [ ! -d ${jobname} ]; then
        echo ./scripts/darknetjs.sh --image_folder ${imagedir} --job_name ${jobname} --clean --noshutdown ${arg[@]} > ${jobname}.log
        ./scripts/darknetjs.sh --image_folder ${imagedir} --job_name ${jobname} --clean --noshutdown ${arg[@]} >> ${jobname}.log 2>&1
    fi
done

if [ -f $HOME/.openrc.sh ]; then
      source $HOME/.openrc.sh
      openstack server shelve `cat /run/cloud-init/.instance-id`
fi