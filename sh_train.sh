export CUDA_VISIBLE_DEVICES=3

python3 run_image_captioning_flax.py \
	--output_dir ./image-captioning-training-results \
	--model_name_or_path ./model \
	--dataset_name ydshieh/coco_dataset_script \
	--dataset_config_name=2017 \
    --data_dir /srv/nas_data1/text/miftah/finetune_huggingface/ppt/flax/stoned \
	--image_column image_path \
	--caption_column caption \
	--do_train --do_eval --do_predict --predict_with_generate \
	--num_train_epochs 100 \
	--eval_steps 50 \
	--logging_steps 50 \
	--learning_rate 3e-5 --warmup_steps 0 \
	--per_device_train_batch_size 32 \
	--per_device_eval_batch_size 32 \
	--overwrite_output_dir \
	--max_target_length 32 \
	--num_beams 8 \
	--preprocessing_num_workers 16 \
	--block_size 16384
