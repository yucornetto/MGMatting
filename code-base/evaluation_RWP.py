import os
import cv2
import numpy as np
from   utils import comput_sad_loss, compute_mse_loss
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-dir', type=str, default='./predRWP/', help="pred alpha dir")
    parser.add_argument('--label-dir', type=str, default='/export/ccvl12b/qihang/MGMatting/data/RealWorldPortrait-636/alpha/', help="GT alpha dir")
    parser.add_argument('--detailmap-dir', type=str, default='/export/ccvl12b/qihang/MGMatting/data/RealWorldPortrait-636/detailmap/', help="trimap dir")

    args = parser.parse_args()

    mse_loss = []
    sad_loss = []

    ### loss_unknown only consider the unknown regions, i.e. trimap==128, as trimap-based methods do
    mse_loss_unknown = []
    sad_loss_unknown = []

    for img in os.listdir(args.label_dir):
        print(img)
        pred = cv2.imread(os.path.join(args.pred_dir, img.replace('.png', '.jpg')), 0).astype(np.float32)
        label = cv2.imread(os.path.join(args.label_dir, img), 0).astype(np.float32)
        detailmap = cv2.imread(os.path.join(args.detailmap_dir, img.replace('.png', '_cloud_trimap.jpg')), 0).astype(np.float32)

        detailmap[detailmap > 0] = 128

        mse_loss_unknown_ = compute_mse_loss(pred, label, detailmap)
        sad_loss_unknown_ = comput_sad_loss(pred, label, detailmap)[0]

        detailmap[...] = 128

        mse_loss_ = compute_mse_loss(pred, label, detailmap)
        sad_loss_ = comput_sad_loss(pred, label, detailmap)[0]

        print('Whole Image: MSE:', mse_loss_, ' SAD:', sad_loss_)
        print('Detail Region: MSE:', mse_loss_unknown_, ' SAD:', sad_loss_unknown_)

        mse_loss_unknown.append(mse_loss_unknown_)
        sad_loss_unknown.append(sad_loss_unknown_)

        mse_loss.append(mse_loss_)
        sad_loss.append(sad_loss_)

    print('Average:')
    print('Whole Image: MSE:', np.array(mse_loss).mean(), ' SAD:', np.array(sad_loss).mean())
    print('Detail Region: MSE:', np.array(mse_loss_unknown).mean(), ' SAD:', np.array(sad_loss_unknown).mean())
