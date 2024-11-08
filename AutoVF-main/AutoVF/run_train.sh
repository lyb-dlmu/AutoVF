python run.py \
    --load_pretrained_model \
    --pretrained_model_name=vqr_pretrained_model.bin \
    --model_name=vqr_fine_tuned_model.bin \
    --output_dir=./saved_models \
    --tokenizer_name=Salesforce/codet5-base \
    --model_name_or_path=Salesforce/codet5-base \
    --do_train \
    --train_data_file=../data/cve_fixes_and_big_vul/train.csv \
    --eval_data_file=../data/cve_fixes_and_big_vul/val.csv \
    --test_data_file=../data/cve_fixes_and_big_vul/test.csv \
    --epochs 75 \
    --encoder_block_size 512 \
    --vul_repair_block_size 256 \
    --train_batch_size 16 \
    --eval_batch_size 2 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log

