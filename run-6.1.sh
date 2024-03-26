export CUBLAS_WORKSPACE_CONFIG=:16:8

python main_testv2.py \
--dataset_dir ./data \
--batch_size 128 \
--epochs 300 \
--lr 0.05 \
--wd 5e-4  \
--lr_scheduler \
--seed 0 \
--alpha 0.2 \
--mixup \
--fig_name lr=0.05-lr_sche-wd=5e-4-mixup.png \
--test
