export CUBLAS_WORKSPACE_CONFIG=:16:8

python main_testv2.py \
--dataset_dir ./data \
--batch_size 128 \
--epochs 300 \
--lr 0.05 \
--wd 0 \
--seed 0 \
--fig_name lr0.05-epoch300-run41.png \
--test