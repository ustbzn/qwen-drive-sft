python "scripts/cache_dataset/fix_mogo_obstacle_frame.py" \
    --logs-dir "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/dataset/navsim_logs/train" \
    --logs-dir "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/dataset/navsim_logs/val" \
    --rebuild-trainval "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/dataset/navsim_logs/trainval" \
    --rebuild-trainval-as-symlink
