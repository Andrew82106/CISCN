import argparse
import math
from io import BytesIO
import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image
from torchvision.transforms.functional import normalize, resize, to_pil_image, to_tensor
from torchcam import methods
from torchcam.utils import overlay_mask
from networks.model import MainNet
from config import num_classes, proposalN, channels

def main(args):

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(args.device)

    # Pretrained imagenet model
    model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)
    save_path = checkpoint_save_path + "best.pt"
    checkpoint = torch.load(save_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model = model.to(device=device)

    # Image
    if args.img.startswith("http"):
        img_path = BytesIO(requests.get(args.img).content)
    else:
        img_path = args.img
    pil_img = Image.open(img_path, mode="r").convert("RGB")
    # Preprocess image
    img_tensor = normalize(to_tensor(resize(pil_img, (320, 320))), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(
        device=device
    )

    if isinstance(args.method, str):
        cam_methods = args.method.split(",")
    else:
        cam_methods = [
            "CAM",
            "GradCAM",
            "GradCAMpp",
            "SmoothGradCAMpp",
            "ScoreCAM",
            "SSCAM",
            "ISCAM",
            "XGradCAM",
            "LayerCAM",
        ]
    # Hook the corresponding layer in the model
    target_layers = [model.pretrained_model.xception_rgb.model.bn4]
    cam_extractors = [methods.__dict__[name](model, target_layer=target_layers, enable_hooks=False) for name in cam_methods]

    # Homogenize number of elements in each row
    num_cols = math.ceil((len(cam_extractors) + 1) / args.rows)
    _, axes = plt.subplots(args.rows, num_cols, figsize=(6, 4))
    # Display input
    ax = axes[0][0] if args.rows > 1 else axes[0] if num_cols > 1 else axes
    ax.imshow(pil_img)
    ax.set_title(args.forgerytype, size=12)

    for idx, extractor in zip(range(1, len(cam_extractors) + 1), cam_extractors):
        extractor._hooks_enabled = True
        model.zero_grad()
        proposalN_windows_score, proposalN_windows_logits, indices, \
        window_scores, scores, srm_image = model(img_tensor.unsqueeze(0), 'test')

        # Select the class index
        class_idx = scores.squeeze(0).argmax().item()
            # if args.class_idx is None else args.class_idx

        # Use the hooked data to compute activation map
        activation_map = extractor(class_idx, scores)[0].squeeze(0).cpu()
        # Visualize the raw CAM
        # plt.imshow(activation_map.squeeze(0).numpy());
        # if num_cols > 1:
        #     for _axes in axes:
        #         if args.rows > 1:
        #             for ax in _axes:
        #                 ax.axis("off")
        #         else:
        #             _axes.axis("off")
        # else:
        #     axes.axis("off")
        # ax = axes[idx // num_cols][idx % num_cols] if args.rows > 1 else axes[idx] if num_cols > 1 else axes
        # ax.set_title("Ours", size=12)
        # # ax.set_title(extractor.__class__.__name__, size=12)

        # Clean data
        extractor.remove_hooks()
        extractor._hooks_enabled = False
        # Convert it to PIL image
        # The indexing below means first image in batch
        heatmap = to_pil_image(activation_map, mode="F")
        # Plot the result
        result = overlay_mask(pil_img, heatmap, alpha=args.alpha)
        ax = axes[idx // num_cols][idx % num_cols] if args.rows > 1 else axes[idx] if num_cols > 1 else axes
        result.save("./1.png")
        ax.imshow(result)
        # ax.set_title(extractor.__class__.__name__, size=12)
        ax.set_title("Ours", size=12)

    # Clear axes
    if num_cols > 1:
        for _axes in axes:
            if args.rows > 1:
                for ax in _axes:
                    ax.axis("off")
            else:
                _axes.axis("off")
    else:
        axes.axis("off")

    plt.tight_layout()
    # if args.savefig:
    #     plt.savefig(args.savefig, dpi=600, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.show(block=not args.noblock)

if __name__ == "__main__":
    model_name = 'tsnet-815'

    training_set = "FF++"
    # FF++ celeb-df-v2
    training_forgery_type = "deepfakes"
    # deepfakes neuraltextures face2face faceshifter faceswap all testdata None
    training_compression = "c23"
    # c23 c40 None
    testing_set = "FF++"
    # FF++ celeb-df-v2
    testing_forgery_type = "deepfakes"
    # deepfakes neuraltextures face2face faceshifter faceswap all testdata None
    testing_compression = "c23"
    # c23 c40 None
    checkpoint_save_path = './output/checkpoint/{0}_{1}({2})--{3}_{4}({5})/{6}/'.format(training_set,
                                                                                        training_forgery_type,
                                                                                        training_compression,
                                                                                        testing_set,
                                                                                        testing_forgery_type,
                                                                                        testing_compression, model_name)
    parser = argparse.ArgumentParser(
        description="Saliency Map comparison", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--img",
        type=str,
        default=r"G:\CISCN作品赛\processeddataset\FF++\\fake\deepfakes\c23\\1.png",
        # deepfakes neuraltextures face2face faceshifter faceswap all testdata None
        help="The image to extract CAM from",
    )
    parser.add_argument("--forgerytype", type=str, default="deepfakes", help="forgery type")
    parser.add_argument("--class-idx", type=int, default=1, help="Index of the class to inspect")
    parser.add_argument("--device", type=str, default=None, help="Default device to perform computation on")
    parser.add_argument("--savefig", type=str, default=r"G:\CISCN作品赛\processeddataset\FF++\fake\deepfakes\c23\1.png", help="Path to save figure")
    parser.add_argument("--method", type=str, default="GradCAM", help="CAM method to use")
    # "CAM", "GradCAM", "GradCAMpp", "SmoothGradCAMpp", "ScoreCAM", "SSCAM", "ISCAM", "XGradCAM", "LayerCAM",
    parser.add_argument("--alpha", type=float, default=0.5, help="Transparency of the heatmap")
    parser.add_argument("--rows", type=int, default=1, help="Number of rows for the layout")
    parser.add_argument("--noblock", dest="noblock", help="Disables blocking visualization", action="store_true")
    args = parser.parse_args()

    main(args)
