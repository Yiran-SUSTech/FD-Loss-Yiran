huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir /mnt/afs/zhengmingkai/zyr/FD-Loss-Yiran/checkpoints/sd-vae-ft-mse

torchrun --nproc_per_node=1 --master_port=29501 eval_all_fds.py \
    --model iMF_XL --tokenizer sdvae --tokenizer_patch_size 8 --patch_size 2 --disable_v_head \
    --cfg 8.0 --cfg_list 8.0 \
    --interval_min 0.42 --interval_max 0.62 \
    --num_sampling_steps 1 \
    --gen_only \
    --eval_ema_labels online \
    --disable_wandb --no_prc \
    --eval_bsz 32 \
    --load_from checkpoints/post-trained/iMF-XL_FD-SIM.pth \
    --vae_path /mnt/afs/zhengmingkai/zyr/FD-Loss-Yiran/checkpoints/sd-vae-ft-mse \
    --class_id_list "$CLASS_LIST" --num_image_per_class 50 \
    --output_dir ./work_dirs/IMF-XL-FDloss_100class_50perclass \
    --project gen_only \
    --exp_name iMF-XL_FD-SIM-class-sample


torchrun --nproc_per_node=1 --master_port=29500 eval_all_fds.py \
    --model JiT_H --rope_2d --learned_pe --legacy_time_convention --ema_type edm \
    --cfg 2.2 --cfg_list 2.2 --interval_min 0.1 --interval_max 1.0 \
    --num_sampling_steps 1 --gen_only --eval_ema_labels online \
    --disable_wandb --no_prc --eval_bsz 1 \
    --load_from checkpoints/post-trained/JiT-H_FD-SIM.pth \
    --class_id_list "$CLASS_LIST" --num_image_per_class 50 \
    --output_dir ./work_dirs/JiT-H-FDloss_100class_50perclass --project gen_only \
    --exp_name JiT-H_FD-SIM-class-sample