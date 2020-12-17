# DEMO
We provide visual samples and comparison with Internet Images here.

## Image Matting Results
We compare MG Matting with other trimap-based/free methods, including [Late Fusion Matting\(LFM\)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_A_Late_Fusion_CNN_for_Digital_Matting_CVPR_2019_paper.pdf), [Deep Image Matting\(DIM\)](https://arxiv.org/pdf/1703.03872.pdf), [Index Matting\(Index\)](https://arxiv.org/pdf/1908.00672.pdf), [GCA Matting\(GCA\)](https://arxiv.org/pdf/2001.04069.pdf), [Context-Aware Matting\(CA\)](https://arxiv.org/pdf/1909.09725.pdf), on Internet images.

Please refer to [IMAGE.md](IMAGE.md) for details.

## Fully Automatic Matting System
**We further train MG Matting with an internal portrait matting dataset consisting of 4395 samples, and combine it with a base segmentation model to obtain a fully automatic matting system**. We compare this system to latest trimap-free matting system [MODNet](https://github.com/ZHKKKe/MODNet) and also commercial software Photoshop 2021.

Please refer to [SYSTEM.md](SYSTEM.md) for details.

## Video Matting
We note that MG Matting, though not utilizing temporal information yet, can potentially produce great results on videos. Please refer to the following links for video demos.

<p align="center">
  <a href="https://youtu.be/CB5pLIbRT28">Video Demo 1</a> |
  <a href="https://youtu.be/ldlvGGWTbFI">Video Demo 2</a> |
  <a href="https://youtu.be/4_4O13yA4AQ">Video Demo 3</a> |
  <a href="https://youtu.be/fOHNvrPwPrE">Video Demo 4</a> 
</p>