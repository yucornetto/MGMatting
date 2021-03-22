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
Please modify the data path in config file (e.g. config/MGMatting-DIM.toml) accordingly, and start training using the following command:
```
bash train.sh
```
or
```
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 python -m torch.distributed.launch --nproc_per_node=4 main.py \
--config=config/MGMatting-DIM.toml
```

### Testing on DIM dataset
```
CUDA_VISIBLE_DEVICES=0 python infer.py --config PATH_TO_CONFIG --checkpoint PATH_TO_CKPT --image-dir PATH_TO_INPUT_IMG --mask-dir PATH_TO_INPUT_MASK --output PATH_TO_SAVE_RESULTS
```

Afterwards, you can evaluate the results by:
```
python evaluation.py --pred-dir PATH_TO_SAVED_RESULTS --label-dir PATH_TO_GROUND_TRUTH --trimap-dir PATH_TO_TRIMAP
```
which will give the MSE/SAD scores under two settings: Whole Image (measured acorss the whole image) and Unknown Only (measured in unknown region indicated by trimap only). Please note that these scores are python reimplmentation, and if you want to report scores in your paper, please use the official matlab codes for evaluation.

### TODO

- Model Zoo providing pretrained models and demo purposes.

- Realworld augmentations.
