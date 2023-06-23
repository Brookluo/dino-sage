#!/bin/bash

rgb_file=/lus/theta-fs0/projects/MultiActiveAI/sage-cloud-data/rgb/processed/rgb_20230521-095736_W057_6144x2048_position10.jpg
thermal_file=/lus/theta-fs0/projects/MultiActiveAI/sage-cloud-data/thermal/20230521-095736_W057_right_336x252_14bit.thermal.celsius_position10.csv
python --version
echo "Getting RGB attention"
python visualize_attention.py --arch vit_tiny --patch_size 16 --image_size 600 800\
   --pretrained_weights /lus/swift/home/brookluo/anl-su23/vicreg-sage/test/vicreg_vit_rgb_left.pt \
   --image_path $rgb_file \
   --output_dir ./rgb --channels=3
echo "Getting thermal attention"
python visualize_attention.py --arch vit_tiny --patch_size 16 --image_size 252 336 \
   --pretrained_weights /lus/swift/home/brookluo/anl-su23/vicreg-sage/test/vicreg_vit_rgb_right.pt \
   --image_path $thermal_file \
   --output_dir ./thermal --channels=1

