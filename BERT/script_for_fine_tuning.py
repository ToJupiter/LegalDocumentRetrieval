!pip install protobuf==3.20.1 sentence accelerate>=0.21.0 datasets faiss-gpu jsonlines numpy
!pip install torch>=1.6.0
!pip install transformers==4.45.2 sentence-transformers==3.1.1 sentencepiece

!python /VeMienTay/finetune.py \
--output_dir /VeMienTay/finetuned \
--model_name_or_path namdp-ptit/ViRanker \
--train_data /VeMienTay/fine_tune_training.json \
--learning_rate 1e-5 \
--num_train_epochs 3 \
--per_device_train_batch_size 16 \
--dataloader_drop_last True \
--normalized True \
--temperature 0.02 \
--query_max_len 64 \
--passage_max_len 256 \
--train_group_size 2 \
--logging_steps 10 \
--query_instruction_for_retrieval "" \
--save_steps 10000
--bf16 True




