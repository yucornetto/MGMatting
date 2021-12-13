import toml
import argparse
import torch
import utils
import networks
from utils import CONFIG

if __name__ == '__main__':
    import onnx
    import onnxruntime as ort
    from onnxsim import simplify

    print('Torch Version: ', torch.__version__, "ONNX Version: ", onnx.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/MGMatting-DIM-100k.toml')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/MGMatting-DIM-100k/latest_model.pth',
                        help="path of checkpoint")
    parser.add_argument('--output_path', type=str, default='./checkpoints/MGMatting-DIM-100k/latest_model.onnx',
                        help="output path")
    parser.add_argument("--dynamic", action="store_true", help="export ONNX with dynamic shape")
    parser.add_argument("--simplify", action="store_true", help="export ONNX with simplify")
    # Parse configuration
    args = parser.parse_args()
    print(args)
    with open(args.config) as f:
        utils.load_config(toml.load(f))

    # Check if toml config file is loaded
    if CONFIG.is_default:
        raise ValueError("No .toml config loaded.")

    # build model
    model = networks.get_generator(encoder=CONFIG.model.arch.encoder, decoder=CONFIG.model.arch.decoder)
    model.cpu()
    print("Generate Model Done.")

    # load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)
    print(f"Load {args.checkpoint} Done.")

    # inference
    model = model.eval()

    image = torch.randn((1, 3, 512, 512), requires_grad=False)
    mask = torch.randn((1, 1, 512, 512), requires_grad=False)
    onnx_file_name = args.output_path
    dynamic_axes = {
        "image": {2: 'height', 3: 'width'},
        "mask": {2: 'height', 3: 'width'},
        "alpha_os1": {2: 'height', 3: 'width'},
        "alpha_os4": {2: 'height', 3: 'width'},
        "alpha_os8": {2: 'height', 3: 'width'},
    }

    # Export the model
    if args.dynamic:
        print('Export the dynamic onnx model ...')
        torch.onnx.export(model,
                          (image, mask),
                          onnx_file_name,
                          export_params=True,
                          opset_version=12,
                          do_constant_folding=True,
                          input_names=["image", "mask"],
                          output_names=["alpha_os1", "alpha_os4", "alpha_os8"],
                          dynamic_axes=dynamic_axes)
    else:
        print('Export the static onnx model ...')
        torch.onnx.export(model,
                          (image, mask),
                          onnx_file_name,
                          export_params=True,
                          opset_version=12,
                          do_constant_folding=True,
                          input_names=["image", "mask"],
                          output_names=["alpha_os1", "alpha_os4", "alpha_os8"])

    print("export onnx done.")
    onnx_model = onnx.load(onnx_file_name)
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))

    if args.dynamic:
        model_simp, check = simplify(onnx_model,
                                     check_n=3,
                                     dynamic_input_shape=True,
                                     input_shapes={"image": [1, 3, 512, 512],
                                                   "mask": [1, 1, 512, 512]}
                                     )
    else:
        model_simp, check = simplify(onnx_model,
                                     check_n=3)

    onnx.save(model_simp, onnx_file_name)
    print(onnx.helper.printable_graph(model_simp.graph))
    print("export onnx sim done.")

    # test onnxruntime
    ort_session = ort.InferenceSession(onnx_file_name)

    for o in ort_session.get_inputs():
        print(o)

    for o in ort_session.get_outputs():
        print(o)

    """
    PYTHONPATH=. python3 ./export_onnx.py --dynamic --simplify --config ./config/MGMatting-DIM-100k.toml --checkpoint ./checkpoints/MGMatting-DIM-100k/latest_model.pth --output_path ./checkpoints/MGMatting-DIM-100k/latest_model.onnx
    """
