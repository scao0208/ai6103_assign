export CUBLAS_WORKSPACE_CONFIG=:16:8

python main.py \
--dataset_dir ./data \
--batch_size 128 \
--epochs 1000 \
--lr 0.05 --wd 0.0005 \
--lr_scheduler \
--mixup \
--seed 0 \
--fig_name lr=0.05-lr_sche-wd=0.0005-mixup.png \
--test
