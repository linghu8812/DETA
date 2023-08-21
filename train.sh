GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/deta_swin_ft.sh --coco_path ./data/coco \
    --finetune weights/adet_swin_pt_o365.pth

GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/deta.sh --coco_path ./data/coco
