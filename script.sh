export CUDA_VISIBLE_DEVICES=0
python -m engine.train --model_save output/models/HS-Pose/ --num_workers 20 --batch_size 16 --train_steps 1500 --seed 1677330429 
python -m evaluation.evaluate  --model_save output/models/HS-Pose/model_149 --resume 1 --resume_model ./output/models/HS-Pose/model_149.pth --eval_seed 1677483078