# Synthesizing Dynamic Patterns by Spatial-Temporal Generative ConvNet
This is a tensorflow implementation of the paper "Synthesizing Dynamic Patterns by Spatial-Temporal Generative ConvNet"

http://www.stat.ucla.edu/~jxie/STGConvNet/STGConvNet.html

## Requirements

We use `tensorflow` (version >= 1.0) in Python 2.7 as the main implementation framwork. We use addtional Python packages for reading and outputing videos. To install, run
```
pip install -r requirements.txt
```

## Guide
Our current release has been tested on Ubuntu 14.04.
### Generating dynamic textures with only temporal stationarity.
Optional arguments for running `main.py`
-  `-sz IMAGE_SIZE`: size of image per frame, default to be 100
-  `-num_epochs NUM_EPOCHS`: number of training iterations
-  `-batch_size BATCH_SIZE`: number of videos for each batch
-  `-sample_batch SAMPLE_BATCH`: number of synthesized results for each batch of training data
-  `-num_frames NUM_FRAMES`: number of frames used in training data
-  `-lr LR`: learning rate
-  `-beta1 BETA1`: momentum1 in Adam
-  `-delta STEP_SIZE`: step size for langevin sampling
-  `-sample_steps SAMPLE_STEPS`: number of sampling steps for langevin dynamics
-  `-output_dir OUTPUT_DIR`: output directory for synthesized results and checkpoints
-  `-category CATEGORY`: target traing class
-  `-data_path DATA_PATH`:  root directory of training data
-  `-log_step LOG_STEP`: number of epochs for each log output 
Usage:
```
python main.py -data_path ./trainingVideo -category fire_pot -output_dir ./output -sample_batch 3
```
The synthesized videos will be stored in `./output/fire_pot/final_results`. To output log information, run
```
tensorboard --log ./output/fire_pot/log
```

# Reference

For any questions, please contact Jianwen Xie (jianwen@ucla.edu) and Zilong Zheng (zilongzheng@cs.ucla.edu).