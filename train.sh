PRE_SEQ_LEN=128
LR=2e-2


export CUDA_VISIBLE_DEVICES=3 
python main.py \
    --do_train \
    --train_file stock/2train.json \
    --validation_file stock/2dev.json \
    --preprocessing_num_workers 10 \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /home/leipeng/ll/model \
    --output_dir output15/adgen-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 4096\
    --max_target_length 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 150 \
    --logging_steps 10 \
    --save_steps 10 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

