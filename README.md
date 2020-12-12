# MG Matting

<p align="center">
  <img src="teaser.png" width="1050" title="Teaser Image"/>
</p>

This repository includes the official project of Mask Guided (MG) Matting, presented in our paper:

**[Mask Guided Matting via Progressive Refinement Network](https://arxiv.org/abs/1908.00672)**

## News
- 12 Dec 2020: Release [Arxiv version of paper](https://arxiv.org/pdf/2011.11961.pdf) and [visualization of images and videos](https://youtu.be/PqJ3BRHX3Lc).


## Highlights
- **Trimap-free:** IndexNet Matting only deals with the upsampling stage but exhibits at least 16.1% relative improvements, compared to the Deep Matting baseline;
- **Foreground Color Prediction:** We predict the foreground color besides alpha matte, we notice and address the inaccuracy in annotated training data by Random Alpha Blending;
- **State-of-the-art Performance:** MG Matting is trained on the public synthetic dataset Composition-1k only without any addtional dataset as other trimap-free methods do, MG Matting can handle high-resolution (e.g. 2k) images achives amazing performance on both synthetic and real-world data;
- **Robust on real-world cases:** This framework also includes our re-implementation of Deep Matting and the pretrained model presented in the Adobe's CVPR17 paper.
   

## Inference
Inference demo shall be released soon...

## Real-world Portrait dataest
The real-world portrait benchmark shall be public avalilable soon. Before that, if you want to test your model or compare with MG Matting, feel free to contact Qihang Yu (yucornetto@gmail.com).


## Citation
If you find this work or code useful for your research, please cite:
```
@inproceedings{hao2019indexnet,
  title={Indices Matter: Learning to Index for Deep Image Matting},
  author={Lu, Hao and Dai, Yutong and Shen, Chunhua and Xu, Songcen},
  booktitle={Proc. IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

## Permission and Disclaimer
This code is only for non-commercial purposes. As covered by the ADOBE IMAGE DATASET LICENSE AGREEMENT, the trained models included in this repository can only be used/distributed for non-commercial purposes. Anyone who violates this rule will be at his/her own risk.
