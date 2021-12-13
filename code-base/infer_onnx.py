import cv2
import argparse
import numpy as np
import torch
import utils
import onnxruntime as ort


def single_inference(onnx_session: ort.InferenceSession, image_dict, post_process=False):
    image, mask = image_dict['image'], image_dict['mask']
    alpha_shape = image_dict['alpha_shape']
    alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = onnx_session.run(
        None, input_feed={"image": image, "mask": mask})
    ### refinement
    alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = \
        torch.from_numpy(alpha_pred_os1), \
        torch.from_numpy(alpha_pred_os4), \
        torch.from_numpy(alpha_pred_os8)

    # alpha_pred = alpha_pred_os1.clone()
    alpha_pred = alpha_pred_os8.clone()
    weight_os4 = utils.get_unknown_tensor_from_pred(alpha_pred,
                                                    rand_width=30,
                                                    train_mode=False)
    alpha_pred[weight_os4 > 0] = alpha_pred_os4[weight_os4 > 0]
    weight_os1 = utils.get_unknown_tensor_from_pred(alpha_pred,
                                                    rand_width=15,
                                                    train_mode=False)
    alpha_pred[weight_os1 > 0] = alpha_pred_os1[weight_os1 > 0]

    h, w = alpha_shape
    alpha_pred = alpha_pred[0, 0, ...].data.cpu().numpy()
    if post_process:
        alpha_pred = utils.postprocess(alpha_pred)
    alpha_pred = alpha_pred * 255
    alpha_pred = alpha_pred.astype(np.uint8)
    alpha_pred = alpha_pred[32:h + 32, 32:w + 32]

    return alpha_pred


def generator_tensor_dict(image_path, mask_path, args):
    # read images
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)

    # to 0~1
    mask = (mask >= args.guidance_thres).astype(np.float32)  ### only keep FG part of trimap

    # mask = mask.astype(np.float32) / 255.0 ### soft trimap

    sample = {'image': image, 'mask': mask, 'alpha_shape': mask.shape}

    # reshape
    h, w = sample["alpha_shape"]

    if h % 32 == 0 and w % 32 == 0:
        padded_image = np.pad(sample['image'], ((32, 32), (32, 32), (0, 0)), mode="reflect")
        padded_mask = np.pad(sample['mask'], ((32, 32), (32, 32)), mode="reflect")
        sample['image'] = padded_image
        sample['mask'] = padded_mask
    else:
        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w
        padded_image = np.pad(sample['image'], ((32, pad_h + 32), (32, pad_w + 32), (0, 0)), mode="reflect")
        padded_mask = np.pad(sample['mask'], ((32, pad_h + 32), (32, pad_w + 32)), mode="reflect")
        sample['image'] = padded_image
        sample['mask'] = padded_mask

    # ImageNet mean & std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    # convert GBR images to RGB
    image, mask = sample['image'][:, :, ::-1], sample['mask']
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

    sample['image'], sample['mask'] = sample['image'].cpu().numpy(), sample['mask'].cpu().numpy()

    print(sample["mask"].min(), sample["mask"].max())  # 0, 1.

    return sample


if __name__ == '__main__':
    print('ONNXRuntime Version: ', ort.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, default='./checkpoints/MGMatting-DIM-100k/latest_model.onnx',
                        help="path of onnx model")
    parser.add_argument('--image_path', type=str, default='./test_input.jpg', help="input image path")
    parser.add_argument('--mask_path', type=str, default='./test_mask.png', help="input mask path")
    parser.add_argument('--output_path', type=str, default='./test_onnx_output.jpg', help="output path")
    parser.add_argument('--guidance-thres', type=int, default=128, help="guidance input threshold")
    parser.add_argument('--post-process', action='store_true', default=False,
                        help='post process to keep the largest connected component')

    # Parse configuration
    args = parser.parse_args()
    print(args)

    onnx_session = ort.InferenceSession(args.onnx)
    print(f"Load {args.onnx} Done!")

    # assume image and mask have the same file name
    print(f'Image: {args.image_path}\n', f'Mask: {args.mask_path}')
    image_dict = generator_tensor_dict(args.image_path, args.mask_path, args)

    alpha_pred = single_inference(onnx_session, image_dict, post_process=args.post_process)

    cv2.imwrite(args.output_path, alpha_pred)
    print(f"Inference ONNX Done! Saved to {args.output_path} !")

    """
    PYTHONPATH=. python3 ./infer_onnx.py --post-process --onnx ./checkpoints/MGMatting-DIM-100k/latest_model.onnx --image_path ./test_input.jpg --mask_path test_mask.png --output_path ./test_onnx_output.jpg 
    """
