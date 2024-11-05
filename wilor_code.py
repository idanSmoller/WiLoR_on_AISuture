from pathlib import Path
import torch
import os
import cv2
import numpy as np

from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from wilor.utils.renderer import Renderer, cam_crop_to_full
from ultralytics import YOLO
from tqdm import tqdm
LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)


def predict_on_folder(input_folder="input", output_folder="output", focal_length=16.8):
    args = {
        "img_folder": input_folder,  # Folder with input images
        "out_folder": output_folder,  # Output folder to save rendered results
        "save_mesh": True,  # If set, save meshes to disk also
        "rescale_factor": 2.0,  # Factor for padding the bbox
        "file_type": ['*.jpg', '*.png', '*.jpeg'],  # List of file extensions to consider
        "focal_length": focal_length  # focal length of the camera that captured the input video
    }

    # Download and load checkpoints
    model, model_cfg = load_wilor(checkpoint_path='./pretrained_models/wilor_final.ckpt',
                                  cfg_path='./pretrained_models/model_config.yaml')

    if args["focal_length"] is not None:
        model_cfg.defrost()
        model_cfg.EXTRA.FOCAL_LENGTH = args["focal_length"]

    detector = YOLO("/strg/E/shared-data/AIxSuture.Tools_n_Hands.yolov8/train4-R_L-Hands/weights/best.pt")
    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    detector = detector.to(device)
    model.eval()

    # Make output directory if it does not exist
    os.makedirs(args["out_folder"], exist_ok=True)

    # Get all demo images ends with .jpg or .png
    img_paths = sorted([img for end in args["file_type"] for img in Path(args["img_folder"]).glob(end)])
    # Iterate over all images in folder
    pbar = tqdm(img_paths)
    for img_path in pbar:
        pbar.set_postfix(img_path=Path(img_path).stem)
        img_cv2 = cv2.imread(str(img_path))
        detections = detector(source=img_path,save=False, conf=0.4, device=[0], verbose=False)[0]
        bboxes = []
        is_right = []
        for det in detections:
            Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
            bboxes.append(Bbox[:4].tolist())

        if len(bboxes) == 0:
            continue
        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args["rescale_factor"])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        all_joints = []
        all_kpts = []

        for batch in dataloader:
            batch = recursive_to(batch, device)

            with torch.no_grad():
                out = model(batch)

            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size,
                                               scaled_focal_length).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))

                verts = out['pred_vertices'][n].detach().cpu().numpy()
                joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()

                is_right = batch['right'][n].cpu().numpy()
                verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
                joints[:, 0] = (2 * is_right - 1) * joints[:, 0]
                cam_t = pred_cam_t_full[n]
                kpts_2d = project_full_img(verts, cam_t, scaled_focal_length, img_size[n])

                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)
                all_joints.append(joints)
                all_kpts.append(kpts_2d)

                # Save all meshes to disk
                if args["save_mesh"]:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_PURPLE, is_right=is_right)
                    tmesh.export(os.path.join(args["out_folder"], f'{img_fn}_{n}.obj'))


def project_full_img(points, cam_trans, focal_length, img_res):
    camera_center = [img_res[0] / 2., img_res[1] / 2.]
    K = torch.eye(3)
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = camera_center[0]
    K[1, 2] = camera_center[1]
    points = points + cam_trans
    points = points / points[..., -1:]

    V_2d = (K @ points.T).T
    return V_2d[..., :-1]
