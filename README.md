# Transformer-For-Geochemical-Anomaly-Detection

This is a PyTorch implementation of the Transformer model for geochemical animaly detection in this paper: 

_An end-to-end Transformer for geochemical anomaly detection._ by Shuyan Yu,Hao Deng*,Zhankun Liu,Jin Chen,Keyan Xiao,Xiancheng Mao
	
## Hardware requirements
- two Nvidia RTX 3090Ti GPUs or higher

## Usage
 1. Model Training
    ```bash
    python train.py -data_pkl ./data/pre_data.pkl -proj_share_weight -label_smoothing -output_dir output -warmup 4000 -epoch 200 -b 8 -save_mode best -use_tb
    ```

2. Geochemical Anomaly Detection

    We use the trained Transformer model for the reconstruction of geochemical data and geochemical anomaly detection. 
    ```bash
    python pred.py -data_pkl ./data/pre_data.pkl -model output/model_best.chkpt
    ```