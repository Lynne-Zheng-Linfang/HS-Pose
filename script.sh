export CUDA_VISIBLE_DEVICES=1
python -m engine.train --model_save output/models/HS-Pose_workers32/ --num_workers 32 --batch_size 16 --train_steps 1500 --seed 1677330429
python -m evaluation.evaluate  --model_save output/models/HS-Pose_workers32/model_149 --resume 1 --resume_model ./output/models/HS-Pose_workers32/model_149.pth --eval_seed 1677483078 --num_workers 32