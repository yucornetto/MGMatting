import os
import cv2
import toml
import argparse
import numpy as np

import torch
from torch.nn import functional as F

import utils
from   utils import CONFIG
import networks


def single_inference(model, image_dict, post_process=False):

    with torch.no_grad():
        image, mask = image_dict['image'], image_dict['mask']
        alpha_shape = image_dict['alpha_shape']
        image = image.cuda()
        mask = mask.cuda()
        pred = model(image, mask)
        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']

        ### refinement
        alpha_pred = alpha_pred_os8.clone().detach()
        weight_os4 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width1, train_mode=False)
        alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4>0]
        weight_os1 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width2, train_mode=False)
        alpha_pred[weight_os1>0] = alpha_pred_os1[weight_os1>0]


        h, w = alpha_shape
        alpha_pred = alpha_pred[0, 0, ...].data.cpu().numpy()
        if post_process:
            alpha_pred = utils.postprocess(alpha_pred)
        alpha_pred = alpha_pred * 255
        alpha_pred = alpha_pred.astype(np.uint8)
        alpha_pred = alpha_pred[32:h+32, 32:w+32]

        return alpha_pred



def generator_tensor_dict(image_path, mask_path, args):
    # read images
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)

    mask = (mask >= args.guidance_thres).astype(np.float32) ### only keep FG part of trimap
    
    #mask = mask.astype(np.float32) / 255.0 ### soft trimap

    sample = {'image': image, 'mask': mask, 'alpha_shape': mask.shape}

    # reshape
    h, w = sample["alpha_shape"]
    
    if h % 32 == 0 and w % 32 == 0:
        padded_image = np.pad(sample['image'], ((32,32), (32, 32), (0,0)), mode="reflect")
        padded_mask = np.pad(sample['mask'], ((32,32), (32, 32)), mode="reflect")
        sample['image'] = padded_image
        sample['mask'] = padded_mask
    else:
        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w
        padded_image = np.pad(sample['image'], ((32,pad_h+32), (32, pad_w+32), (0,0)), mode="reflect")
        padded_mask = np.pad(sample['mask'], ((32,pad_h+32), (32, pad_w+32)), mode="reflect")
        sample['image'] = padded_image
        sample['mask'] = padded_mask

    # ImageNet mean & std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    # convert GBR images to RGB
    image, mask = sample['image'][:,:,::-1], sample['mask']
    # swap color axis
    image = image.transpose((2, 0, 1)).astype(np.float32)

    mask = np.expand_dims(mask.astype(np.float32), axis=0)

    # normalize image
    image /= 255.

    # to tensor
    sample['image'], sample['mask'] = torch.from_numpy(image), torch.from_numpy(mask)
    sample['image'] = sample['image'].sub_(mean).div_(std)

    # add first channel
    sample['image'], sample['mask'] = sample['image'][None, ...], sample['mask'][None, ...]

    return sample

if __name__ == '__main__':
    print('Torch Version: ', torch.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/MGMatting-DIM-100k.toml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/MGMatting-DIM-100k/latest_model.pth',
                        help="path of checkpoint")
    parser.add_argument('--image-dir', type=str, default='/export/ccvl12b/qihang/MGMatting/data/Combined_Dataset/Test_set/merged/', help="input image dir")
    parser.add_argument('--mask-dir', type=str, default='/export/ccvl12b/qihang/MGMatting/data/Combined_Dataset/Test_set/trimaps/', help="input mask dir")
    parser.add_argument('--image-ext', type=str, default='.png', help="input image ext")
    parser.add_argument('--mask-ext', type=str, default='.png', help="input mask ext")
    parser.add_argument('--output', type=str, default='predDIM/', help="output dir")
    parser.add_argument('--guidance-thres', type=int, default=128, help="guidance input threshold")
    parser.add_argument('--post-process', action='store_true', default=False, help='post process to keep the largest connected component')
    
    # Parse configuration
    args = parser.parse_args()
    with open(args.config) as f:
        utils.load_config(toml.load(f))

    # Check if toml config file is loaded
    if CONFIG.is_default:
        raise ValueError("No .toml config loaded.")

    args.output = os.path.join(args.output, CONFIG.version+'_'+args.checkpoint.split('/')[-1])
    utils.make_dir(args.output)

    # build model
    model = networks.get_generator(encoder=CONFIG.model.arch.encoder, decoder=CONFIG.model.arch.decoder)
    model.cuda()

    # load checkpoint
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)

    # inference
    model = model.eval()

    for image_name in os.listdir(args.image_dir):
        # assume image and mask have the same file name
        image_path = os.path.join(args.image_dir, image_name)
        mask_path = os.path.join(args.mask_dir, image_name.replace(args.image_ext, args.mask_ext))
        print('Image: ', image_path, ' Mask: ', mask_path)
        image_dict = generator_tensor_dict(image_path, mask_path, args)

        alpha_pred = single_inference(model, image_dict, post_process=args.post_process)

        cv2.imwrite(os.path.join(args.output, image_name), alpha_pred)
