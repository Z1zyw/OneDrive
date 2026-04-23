# Train & inference
## Train
You can train the model following:

```bash
bash launch_train.sh projects/configs/OneDrive/stage1_perception_pretrain.py 8 8
bash launch_train.sh projects/configs/OneDrive/stage2_planning_adaptation.py 8 8
bash launch_train.sh projects/configs/OneDrive/stage3_joint_training.py 8 8
```
Before running Stage 2 and Stage 3, you need to specify the checkpoint from Stage 1/2.
Please modify the following config files:
- stage2_planning_adaptation.py
- stage3_joint_training.py
```bash
load_from = "----YOUR Stage1/2 CKPT----"
```


## Evaluation
**1. OpenLoop Planning**
Run the evaluation command
```bash
bash launch_test.sh projects/configs/OneDrive/stage3_joint_training.py {Your Stage3 Checkpoint} 8 
```
