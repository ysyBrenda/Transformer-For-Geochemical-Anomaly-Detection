# Transformer-For-Geochemical-Anomaly-Detection

This is a PyTorch implementation of the Transformer model for geochemical animaly detection in this paper: 

_An end-to-end Transformer for geochemical anomaly detection._ by Shuyan Yu,Hao Deng*,Zhankun Liu,Jin Chen,Keyan Xiao,Xiancheng Mao
	
## Hardware requirements
- two Nvidia RTX 3090Ti GPUs or higher
- 
## Dependencies required

> + Ubuntu 16.04
> + Python 3.7
> + Pytorch 1.3.0
> + dill 0.3.3
> + tqdm 4.64.0

## Usage
 1. Data preprocessing

    run `process_data.py` to generate pkl files.

 2. Model Training
    ```bash
    python train.py -data_pkl ./data/pre_data.pkl -output_dir output -n_head 2 -n_layer 4 -warmup 128000 -lr_mul 200 -epoch 50 -b 8 -save_mode best -use_tb -seed 10 -unmask 0.3 -T 2 -isRandMask -isContrastLoss
    ```
    You can use the `gridsearch.sh` to find the optimal parameters.
  

3. Geochemical Anomaly Detection
    
    We use the trained Transformer model for the reconstruction of geochemical data and geochemical anomaly detection. 
     ```bash
     python anomaly_detection.py -data_pkl ./data/pre_data.pkl -model output/model_best.chkpt -raw_data ./data/pos_feature.csv -Au_data ./data/Au_data.csv
     ```
### Data

The data you need to prepare are:

    1. geochemical data, including coordinates and elemental concentration values (pos_feature.csv)
    2. the coordinates of known mine sites. (Au.csv)

Put the above data into the `data` folder in csv format.

---
# Acknowledgement

- The implementation borrows heavily from [attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch) in some parts of the Transformer's architecture.
