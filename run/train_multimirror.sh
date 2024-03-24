CUDA_VISIBLE_DEVICES=0 python -m src.train \
    --train_data "./data/fa/train.tsv" \
    --output_dir "./output" \
    --epoch 10 \
    --batch_size 16
