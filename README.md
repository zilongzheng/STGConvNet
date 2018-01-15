# Synthesizing Dynamic Patterns by Spatial-Temporal Generative ConvNet
This repository contains a tensorflow implementation for the paper "[Synthesizing Dynamic Patterns by Spatial-Temporal Generative ConvNet](http://www.stat.ucla.edu/~jxie/STGConvNet/STGConvNet_file/doc/STGConvNet.pdf)".

http://www.stat.ucla.edu/~jxie/STGConvNet/STGConvNet.html

![fire_pot](result/sample.gif)

## Requirements

We use `tensorflow` (version >= 1.0) in Python 2.7 as the main implementation framework. We use additional Python
packages for reading and saving images. To install, run
```
pip install -r requirements.txt
```
We also use `ffmpeg` for reading and generating videos. To install, run
```
sudo sh ffmpeg_installer.sh
```

## Training
Our current code has been tested on Ubuntu 14.04. Currently this repository only contains `fire_pot` in `trainingVideo`
for the testing purpose. Other training videos can be downloaded [here](http://www.stat.ucla.edu/~jxie/STGConvNet/STGConvNet.html).

We contains three different descriptor network in our `model.py`.

| model      | description |
|------------| ----------------------------------------------------|
|     ST     | Spatial-temporal model for generating dynamic textures with both spatial and temporal stationarity (Exp 1, image size 100)|
|    FC_S    | Spatial fully-connected model for generating  dynamic textures with only temporal stationarity (Exp 2, image size 100) |
| FC_S_large | Spatial fully-connected model for generating  dynamic textures with only temporal stationarity (Exp 2, image size 224, as described in paper) |

To synthesizing 3 different outputs of fire pot with single video input(top 70 frames):
```
python main.py -data_path ./trainingVideo -category fire_pot -output_dir ./output -sz 224 -num_chain 3 -batch_size 1 -lr 0.001 -num_frames 70
```
The synthesized videos will be stored in `./output/fire_pot/final_results` and images for each frame will be stored in
`./output/fire_pot/synthesized_sequence`.

To output log information, run
```
tensorboard --log ./output/fire_pot/log
```

## Reference

    @article{Xie2017,
        author = {Jianwen Xie and Song-Chun Zhu and Yingnian Wu},
        title = {Synthesizing Dynamic Patterns by Spatial-Temporal Generative ConvNet},
        journal = {arXiv preprint arXiv:1606.00972},
        year = {2016}
    }

For any questions, please contact Jianwen Xie (jianwen@ucla.edu) and Zilong Zheng (zilongzheng@cs.ucla.edu).