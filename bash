Pretrain:
python3 src/run.py --seed 42 --argoverse --master_port 8240 --future_frame_num 30 --do_train \
--data_dir /your/path/to/pretrain/dataset/ \
--num_train_epochs 5.0 --learning_rate 0.0001 --train_batch_size 256 \
--output_dir /your/output/dir --hidden_size 128 --use_map --core_num 16 --use_centerline --distributed_training 1 \
--other_params semantic_lane direction l1_loss goals_2D enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph \
lane_scoring complete_traj complete_traj-3 pretrain=0.50 mask='mix' pt_visualize

# For PT evaluation, add 'pt_visualize' in other_params.

Finetune:
python3 src/run.py --seed 42 --argoverse --master_port 8248 --future_frame_num 30 --do_train \
--data_dir /your/path/to/finetune/data/ \
--num_train_epochs 16.0 --learning_rate 0.001 --train_batch_size 64 \
--output_dir /your/output/dir --hidden_size 128 --use_map --core_num 16 --use_centerline --distributed_training 1 \
--other_params semantic_lane direction l1_loss goals_2D enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph \
lane_scoring complete_traj complete_traj-3 use_pt_model=5

# Please either keep your output dir same, or put pretrain model in the /your/output/dir for the fine-tuning, so that the fintuning can read that file

Eval
python3 src/run.py --seed 42 --argoverse --master_port 8249 --future_frame_num 30 --do_train \
--data_dir_for_val /your/path/to/val/data/ \
--model_recover_path /your/path/to/model/model.16.bin \
--output_dir /your/path/to/output/dir --hidden_size 128 --train_batch_size 64 --use_map --core_num 32 --use_centerline --distributed_training 1 \
--do_eval --eval_params optimization MRminFDE=0.0 cnt_sample=9 opti_time=0.1 \
--other_params semantic_lane direction l1_loss goals_2D enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph \
lane_scoring complete_traj complete_traj-3