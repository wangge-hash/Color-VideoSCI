# High-fidelity color video snapshot compressive imaging system
This repo is the implementation of [High-fidelity color video snapshot compressive imaging system](https://www.sciencedirect.com/science/article/pii/S0030399226008534).


## Installation
```
pip install -r requirements.txt
```

## Training 
Support multi GPUs and single GPU training efficiently. First download DAVIS 2017 dataset from [DAVIS website](https://davischallenge.org/), then modify *data_root* value in *configs/\_base_/davis.py* file, make sure *data_root* link to your training dataset path.

Launch multi GPU training by the statement below:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train_color_fuse.py configs/EfficientSCI/efficientsci_base.py --distributed=True
```

Launch single GPU training by the statement below.

Default using GPU 0. One can also choosing GPUs by specify CUDA_VISIBLE_DEVICES

```
python tools/train_color_fuse.py configs/EfficientSCI/efficientsci_base.py
```

## Testing on Simulation Dataset 
The testing procedures and model evaluation process for this project have been organized in a Jupyter Notebook. Open and run the file `eval_color_sim.ipynb` in your environment; execute the code cells sequentially to view the full testing workflow and results. Checkpoints for base model (ckpt0_color.pth) and our model (ckpt1_color.pth) are in [BaiduNetdisk](https://pan.baidu.com/s/1k19pmeF_e9u3TihaFr-CdA?pwd=u6en), and place them in the checkpoints folder.



## Citation

```
@article{LIU2026115502,
title = {High-fidelity color video snapshot compressive imaging system},
journal = {Optics & Laser Technology},
volume = {203},
pages = {115502},
year = {2026},
issn = {0030-3992},
doi = {https://doi.org/10.1016/j.optlastec.2026.115502},
url = {https://www.sciencedirect.com/science/article/pii/S0030399226008534},
author = {Xing Liu and Ge Wang and Yihang Zhai and Panpan Cheng and Yunfeng Song and Mengyuan Liu and Xin Yuan},
}
```
