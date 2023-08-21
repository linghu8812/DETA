./configs/deta.sh --eval --coco_path ./data/coco --resume weights/adet_2x_checkpoint0023.pth

GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/deta.sh \
    --eval --coco_path ./data/coco --resume weights/adet_2x_checkpoint0023.pth

GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/deta_swin_ft.sh \
    --eval --coco_path ./data/coco --resume weights/adet_swin_ft.pth
