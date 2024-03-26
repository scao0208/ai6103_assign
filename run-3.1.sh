export CUBLAS_WORKSPACE_CONFIG=:16:8

python main_testv2.py \
--dataset_dir ./data \
--batch_size 128 \
--epochs 15 \
--lr 0.5 \
--wd 0 \
--seed 0 \
--fig_name lr0.5-epoch15.png \
--test