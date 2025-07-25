# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it
# under the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

'''
detector for single person detection:
    https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox

tracker for multi-person tracking and ReID:
    https://github.com/open-mmlab/mmtracking/tree/master/configs/mot/bytetrack
'''

import os
import os.path as osp
import cv2
import copy
import glob
import argparse
import numpy as np
from tqdm import tqdm
import pickle
import joblib
import torch
import torchgeometry as tgm
from torch.utils.data import DataLoader

import smplx

from models.smpl import SMPL
from models.cliff_hr48.cliff import CLIFF as cliff_hr48
from models.cliff_res50.cliff import CLIFF as cliff_res50

from common import constants
from common.renderer_pyrd import Renderer
from common.mocap_dataset import MocapDataset
from common.utils import estimate_focal_length
from common.utils import strip_prefix_if_present, cam_crop2full, video_to_images

import mmcv

from mmdet.apis import inference_detector, init_detector


def perspective_projection(points, rotation, translation, focal_length,
                           camera_center):
    """This function computes the perspective projection of a set of points.

    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)

    # ATTENTION: the line shoule be commented out as the points have been aligned
    # points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


def main(args):
    
    device = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available() else torch.device('cpu')

    print("Input path:", args.input_path)
    print("Input type:", args.input_type)
    if args.input_type == "image":
        img_path_list = [args.input_path]
        base_dir = osp.dirname(osp.abspath(args.input_path))
        front_view_dir = side_view_dir = bbox_dir = base_dir
        result_filepath = f"{args.input_path[:-4]}_cliff_{args.backbone}.npz"
    else:
        if args.input_type == "video":
            basename = osp.basename(args.input_path).split('.')[0]
            base_dir = osp.join(osp.dirname(osp.abspath(args.input_path)), basename)
            img_dir = osp.join(base_dir, "imgs")
            front_view_dir = osp.join(base_dir, "front_view_%s" % args.backbone)
            side_view_dir = osp.join(base_dir, "side_view_%s" % args.backbone)
            bbox_dir = osp.join(base_dir, "bbox")
            result_filedir_path = osp.join(base_dir)
            if osp.exists(img_dir):
                print(f"Skip extracting images from video, because \"{img_dir}\" already exists")
            else:
                os.makedirs(img_dir, exist_ok=True)
                video_to_images(args.input_path, img_folder=img_dir)

        elif args.input_type == "folder":
            img_dir = osp.join(args.input_path, "imgs")
            front_view_dir = osp.join(args.input_path, "front_view_%s" % args.backbone)
            side_view_dir = osp.join(args.input_path, "side_view_%s" % args.backbone)
            bbox_dir = osp.join(args.input_path, "bbox")
            basename = args.input_path.split('/')[-1]
            result_filedir_path = osp.join(args.input_path)

        # get all image paths
        img_path_list = glob.glob(osp.join(img_dir, '*.jpg'))
        img_path_list.extend(glob.glob(osp.join(img_dir, '*.png')))
        img_path_list.sort()
        img_names = [osp.basename(img_path) for img_path in img_path_list]
    # load all images
    print("Loading images ...")
    orig_img_bgr_all = [cv2.imread(img_path) for img_path in tqdm(img_path_list)]
    print("Image number:", len(img_path_list))
    
    # 모든 이미지를 정사각형으로 만들기 위해 패딩 추가
    padded_img_bgr_all = []
    for img in orig_img_bgr_all:
        h, w = img.shape[:2]
        max_size = max(h, w)
        if h != w:
            # 흰색 배경의 정사각형 이미지 생성 
            padded_img = np.ones((max_size, max_size, 3), dtype=np.uint8) * 255
            
            # 원본 이미지를 중앙에 배치
            y_offset = (max_size - h) // 2
            x_offset = (max_size - w) // 2
            padded_img[y_offset:y_offset+h, x_offset:x_offset+w] = img
            padded_img = cv2.resize(padded_img, (512, 512))
        else:
            padded_img = cv2.resize(img, (512, 512))
        padded_img_bgr_all.append(padded_img)
    
    # 패딩된 이미지 저장
    padded_imgs_dir = osp.join(args.input_path, 'padded_imgs')
    os.makedirs(padded_imgs_dir, exist_ok=True)
    
    for i, padded_img in enumerate(padded_img_bgr_all):
        img_name = img_names[i][:-4]
        filename = f'{img_name}.png'
        save_path = osp.join(padded_imgs_dir, filename)
        cv2.imwrite(save_path, padded_img)
        
    orig_img_bgr_all = padded_img_bgr_all
    # get all image paths
    img_path_list = glob.glob(osp.join(padded_imgs_dir, '*.jpg'))
    img_path_list.extend(glob.glob(osp.join(padded_imgs_dir, '*.png')))
    img_path_list.sort()
    
    from mmdet.apis import DetInferencer
    # https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox
    config = './mmdetection/configs/yolox/yolox_x_8x8_300e_coco.py'
    checkpoint = '/home/oem/members/dyub/CLIFF/data/ckpt/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'
    
    # load detector
    model = DetInferencer(model='yolox_x_8x8_300e_coco', weights=checkpoint, device=device)
    
    # save results
    detection_all = []
    
    # mmdetection procedure
    imgs = img_path_list
    
    # only take-out person (id=0)
    class_id = 0
    for i, img in enumerate(imgs):
        result = model(img, out_dir='outputs_det/', no_save_pred=False)['predictions'][0]
        
        if len(result['labels']) == 0:
            continue
        x1, y1, x2, y2 = result['bboxes'][0]
        score = result['scores'][0]        
        if result['labels'][0] != 0:
            continue

        detection_all.append([i, x1, y1, x2, y2, score, 0.99, 0])
        
    # list to array
    detection_all = np.array(detection_all)
    print("--------------------------- 3D HPS estimation ---------------------------")
    # Create the model instance
    cliff = eval("cliff_" + args.backbone)
    cliff_model = cliff(constants.SMPL_MEAN_PARAMS).to(device)
    # Load the pretrained model
    print("Load the CLIFF checkpoint from path:", args.ckpt)
    state_dict = torch.load(args.ckpt)['model']
    state_dict = strip_prefix_if_present(state_dict, prefix="module.")
    cliff_model.load_state_dict(state_dict, strict=True)
    cliff_model.eval()

    # Setup the SMPL model
    smpl_model = SMPL(constants.SMPL_MODEL_DIR).to(device)

    pred_vert_arr = []
    if args.save_results:
        smpl_pose = []
        smpl_betas = []
        smpl_trans = []
        smpl_joints = []
        cam_focal_l = []
    mocap_db = MocapDataset(orig_img_bgr_all, detection_all)
    mocap_data_loader = DataLoader(mocap_db, batch_size=min(args.batch_size, len(detection_all)), num_workers=0)
    for batch in tqdm(mocap_data_loader):
        norm_img = batch["norm_img"].to(device).float()
        center = batch["center"].to(device).float()
        scale = batch["scale"].to(device).float()
        img_h = batch["img_h"].to(device).float()
        img_w = batch["img_w"].to(device).float()
        focal_length = batch["focal_length"].to(device).float()

        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
        bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]
        
        with torch.no_grad():
            pred_rotmat, pred_betas, pred_cam_crop = cliff_model(norm_img, bbox_info)

        # convert the camera parameters from the crop camera to the full camera
        full_img_shape = torch.stack((img_h, img_w), dim=-1)
        pred_cam_full = cam_crop2full(pred_cam_crop, center, scale, full_img_shape, focal_length)

        pred_output = smpl_model(betas=pred_betas,
                                 body_pose=pred_rotmat[:, 1:],
                                 global_orient=pred_rotmat[:, [0]],
                                 pose2rot=False,
                                 transl=pred_cam_full)
        pred_vertices = pred_output.vertices
        pred_vert_arr.extend(pred_vertices.cpu().numpy())
        
        pred_tpose = smpl_model(betas=pred_betas,
                                 body_pose=torch.zeros_like(pred_rotmat[:, 1:]),
                                 global_orient=torch.zeros_like(pred_rotmat[:, [0]]),
                                 pose2rot=False,
                                 transl=torch.zeros_like(pred_cam_full))
        pred_tpose_joints = pred_tpose.joints[:, :24, :]
        pred_tpose_root = pred_tpose_joints[:, [0], :]
        
        # re-project to 2D keypoints on image plane for calculating reprojection loss
        '''
        # visualize
        for index, (px, py) in enumerate(pred_keypoints2d[0]):
            cv2.circle(img, (int(px), int(py)), 1, [255, 128, 0], 2)
        cv2.imwrite("front_view_kpt.jpg", img)
        '''
        pred_joints = pred_output.joints[:,:24,:]
        camera_center = torch.hstack((img_w[:,None], img_h[:,None])) / 2
        pred_keypoints2d = perspective_projection(
                pred_joints,
                rotation=torch.eye(3, device=device).unsqueeze(0).expand(pred_joints.shape[0], -1, -1),
                translation=pred_cam_full,
                focal_length=focal_length,
                camera_center=camera_center)
        
        if args.save_results:
            # default pose_format is rotation matrix instead of axis-angle
            if args.pose_format == "aa":
                rot_pad = torch.tensor([0, 0, 1], dtype=torch.float32, device=device).view(1, 3, 1)
                rot_pad = rot_pad.expand(pred_rotmat.shape[0] * 24, -1, -1)
                rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad), dim=-1)
                pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)  # N*72
            else:
                pred_pose = pred_rotmat  # N*24*3*3

            smpl_pose.extend(pred_pose.cpu().numpy())
            smpl_betas.extend(pred_betas.cpu().numpy())
            smpl_trans.extend(pred_cam_full.cpu().numpy())
            smpl_joints.extend(pred_output.joints.cpu().numpy()[:, :24, :])
            cam_focal_l.extend(focal_length.cpu().numpy())
    
    bboxes = detection_all[:, 1:5]
    
    if args.save_results:
        mesh_infos = {}
        cameras = {}
        for i, img_name in enumerate(img_names):
            namekey = img_name[:-4]

            extrinsics = np.eye(4)
            intrinsics = np.eye(3)
            
            intrinsics[0, 0] = focal_length[i].item()
            intrinsics[1, 1] = focal_length[i].item()
            intrinsics[0, 2] = img_w[i].item() / 2
            intrinsics[1, 2] = img_h[i].item() / 2
            cameras[namekey] = {
                'extrinsics': extrinsics,
                'intrinsics': intrinsics,
            }

            mesh_infos[namekey] = {}
            
            Rh = smpl_pose[i][:3]
            Rh_mat = cv2.Rodrigues(Rh)[0]
            tpose_root = pred_tpose_root[i].cpu().numpy().ravel()
            Th = tpose_root - np.matmul(tpose_root, Rh_mat.T) + smpl_trans[i]
            
            poses = np.zeros((72,), dtype=np.float32)
            poses[3:72] = smpl_pose[i][3:]
            
            mesh_infos[namekey]['Rh'] = Rh_mat
            mesh_infos[namekey]['Th'] = Th
            mesh_infos[namekey]['poses'] = poses
            mesh_infos[namekey]['joints'] = smpl_joints[i]
            mesh_infos[namekey]['tpose_joints'] = pred_tpose_joints[i].cpu().numpy()
            mesh_infos[namekey]['betas'] = smpl_betas[i]

        joblib.dump(mesh_infos, osp.join(result_filedir_path, f"mesh_infos.joblib"))
        joblib.dump(cameras, osp.join(result_filedir_path, f"cameras.joblib"))

        print(f"Save results to \"{result_filedir_path}\"")

    print("--------------------------- Visualization ---------------------------")
    # make the output directory
    os.makedirs(front_view_dir, exist_ok=True)
    print("Front view directory:", front_view_dir)
    if args.show_sideView:
        os.makedirs(side_view_dir, exist_ok=True)
        print("Side view directory:", side_view_dir)
    if args.show_bbox:
        os.makedirs(bbox_dir, exist_ok=True)
        print("Bounding box directory:", bbox_dir)

    pred_vert_arr = np.array(pred_vert_arr)
    for img_idx, orig_img_bgr in enumerate(tqdm(orig_img_bgr_all)):
        chosen_mask = detection_all[:, 0] == img_idx
        chosen_vert_arr = pred_vert_arr[chosen_mask]
        
        # setup renderer for visualization
        img_h, img_w, _ = orig_img_bgr.shape
        focal_length = estimate_focal_length(img_h, img_w)
        
        renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                            faces=smpl_model.faces,
                            same_mesh_color=False)
        front_view = renderer.render_front_view(chosen_vert_arr,
                                                bg_img_rgb=orig_img_bgr[:, :, ::-1].copy())

        # save rendering results
        basename = osp.basename(img_path_list[img_idx]).split(".")[0]
        filename = basename + "_front_view_cliff_%s.jpg" % args.backbone
        front_view_path = osp.join(front_view_dir, filename)
        cv2.imwrite(front_view_path, front_view[:, :, ::-1])

        if args.show_sideView:
            side_view_img = renderer.render_side_view(chosen_vert_arr)
            filename = basename + "_side_view_cliff_%s.jpg" % args.backbone
            side_view_path = osp.join(side_view_dir, filename)
            cv2.imwrite(side_view_path, side_view_img[:, :, ::-1])

        # delete the renderer for preparing a new one
        renderer.delete()

        # draw the detection bounding boxes
        if args.show_bbox:
            chosen_detection = detection_all[chosen_mask]
            bbox_info = chosen_detection[:, 1:6]

            bbox_img_bgr = orig_img_bgr.copy()
            for min_x, min_y, max_x, max_y, conf in bbox_info:
                
                if conf == 0:
                    continue
                
                ul = (int(min_x), int(min_y))
                br = (int(max_x), int(max_y))
                cv2.rectangle(bbox_img_bgr, ul, br, color=(0, 255, 0), thickness=2)
                cv2.putText(bbox_img_bgr, "%.1f" % conf, ul,
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1.0, color=(0, 0, 255), thickness=1)
            filename = basename + "_bbox.jpg"
            bbox_path = osp.join(bbox_dir, filename)
            cv2.imwrite(bbox_path, bbox_img_bgr)

    # make videos
    if args.make_video:
        print("--------------------------- Making videos ---------------------------")
        from common.utils import images_to_video
        images_to_video(front_view_dir, video_path=front_view_dir + ".mp4", frame_rate=args.frame_rate)
        if args.show_sideView:
            images_to_video(side_view_dir, video_path=side_view_dir + ".mp4", frame_rate=args.frame_rate)
        if args.show_bbox:
            images_to_video(bbox_dir, video_path=bbox_dir + ".mp4", frame_rate=args.frame_rate)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_type', default='image', choices=['image', 'folder', 'video'],
                        help='input type')
    parser.add_argument('--input_path', default='examples_image/im00025.png', help='path to the input data')

    parser.add_argument('--ckpt',
                        default="data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt",
                        help='path to the pretrained checkpoint')
    parser.add_argument("--backbone", default="hr48", choices=['res50', 'hr48'],
                        help="the backbone architecture")
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for detection and motion capture')

    parser.add_argument('--save_results', action='store_true',
                        help='save the results as a npz file')
    parser.add_argument('--pose_format', default='aa', choices=['aa', 'rotmat'],
                        help='aa for axis angle, rotmat for rotation matrix')

    parser.add_argument('--show_bbox', action='store_true',
                        help='show the detection bounding boxes')
    parser.add_argument('--show_sideView', action='store_true',
                        help='show the result from the side view')

    parser.add_argument('--make_video', action='store_true',
                        help='make a video of the rendering results')
    parser.add_argument('--frame_rate', type=int, default=30, help='frame rate')
    
    # NEW!
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--multi', action='store_true', help='multi-person')
    parser.add_argument('--infill', action='store_true', help='motion interpolation, only support linear interpolation now')
    parser.add_argument('--smooth', action='store_true', help='motion smooth, support oneeuro, gaus1d, savgol, smoothnet')
    
    args = parser.parse_args()
    main(args)
