# Mask Guided Matting via Progressive Refinement Network
Code and models for the paper [Mask Guided Matting via Progressive Refinement Network](https://arxiv.org/abs/2012.06722) (CVPR 2021).

### Requirements
#### Packages:
- torch >= 1.1
- tensorboardX
- numpy
- opencv-python
- toml
- easydict
- pprint

For ImageNet pretrained weight and DIM dataset preparation, please refer to [GCA-Matting](https://github.com/Yaoyi-Li/GCA-Matting).

### Training on DIM dataset
Please modify the data path in config file (e.g. config/MGMatting-DIM-100k.toml) accordingly, and start training using the following command:
```
bash train.sh
```
or
```
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 python -m torch.distributed.launch --nproc_per_node=4 main.py \
--config=config/MGMatting-DIM-100k.toml
```

### Testing on DIM dataset
```
CUDA_VISIBLE_DEVICES=0 python infer.py --config PATH_TO_CONFIG --checkpoint PATH_TO_CKPT --image-dir PATH_TO_INPUT_IMG --mask-dir PATH_TO_INPUT_MASK --output PATH_TO_SAVE_RESULTS --guidance-thres 170
```

Afterwards, you can evaluate the results by:
```
python evaluation.py --pred-dir PATH_TO_SAVED_RESULTS --label-dir PATH_TO_GROUND_TRUTH --trimap-dir PATH_TO_TRIMAP
```
which will give the MSE/SAD scores under two settings: Whole Image (measured acorss the whole image) and Unknown Only (measured in unknown region indicated by trimap only). Please note that these scores are python reimplmentation, and if you want to report scores in your paper, please use the official matlab codes for evaluation.

### Training on DIM dataset for RWP benchmark
**Please note that we exclude the transparent objects from DIM training set for a better generalization to real-world portrait cases. You can refer to /utils/copy_data.py for details about preparing the training set.** Afterwards, you can start training using the following command:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 python -m torch.distributed.launch --nproc_per_node=4 main.py \
--config=config/MGMatting-RWP-100k.toml
```

### Testing on RWP benchmark
```
CUDA_VISIBLE_DEVICES=0 python infer.py --config PATH_TO_CONFIG --checkpoint PATH_TO_CKPT --image-dir PATH_TO_INPUT_IMG --mask-dir PATH_TO_INPUT_MASK --output PATH_TO_SAVE_RESULTS --image-ext .jpg --mask-ext .png --guidance-thres 128 --post-process
```

Afterwards, you can evaluate the results by:
```
python evaluation_RWP.py --pred-dir PATH_TO_SAVED_RESULTS --label-dir PATH_TO_GROUND_TRUTH --detailmap-dir PATH_TO_DETAILMAP
```
which will give the MSE/SAD scores under two settings: Whole Image (measured acorss the whole image) and Detail Only (measured in detail region indicated by detail map only).


### Model Zoo
  | DIM dataset                             | MSE | SAD | Grad | Conn |
  |------------------------------| -------------| -------------| -------------| -------------|
  |[MGMatting-DIM-100k](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/qyu13_jh_edu/EblaRdwYHr1Esqbe48HfMT8Bl1y6n1PducNZ2ml3DqSAaw?e=vfbfT0)       | 7.18     |  31.76    | 13.41     |  27.83    |

  | RWP dataset                             | MSE<sub>WholeImage</sub> | SAD<sub>WholeImage</sub> | MSE<sub>Detail</sub> | SAD<sub>Detail</sub> |
  |------------------------------| -------------| -------------| -------------| -------------|
  |[MGMatting-RWP-100k](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/qyu13_jh_edu/Edl8x0nQjy1JhGP6rcV0N-cB654HpmZZa5bwW9rYUvmsJg?e=J3lSba)       | 9.39     |  28.64    | 55.57     |  16.95    |


### TODO

- Foreground prediction and random alpha blending (RAB).
