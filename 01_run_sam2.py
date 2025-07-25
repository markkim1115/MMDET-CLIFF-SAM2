import joblib
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import cv2
import os
from tqdm import tqdm

# ==================== 설정 옵션 ====================
MIN_MASK_AREA = 100  # 최소 마스크 영역 (픽셀 수)
MAX_MASKS = 50  # 최대 마스크 개수 (신뢰도 무관하게 모든 마스크 저장)
# ===================================================

def extract_bbox_from_mask(mask):
    """마스크에서 바운딩 박스를 추출합니다."""
    y_indices, x_indices = np.where(mask)
    
    if len(y_indices) == 0:
        return None
    
    x_min, x_max = int(x_indices.min()), int(x_indices.max())
    y_min, y_max = int(y_indices.min()), int(y_indices.max())
    
    return [x_min, y_min, x_max, y_max]

def create_mask_visualization(img, mask):
    """마스크를 시각화합니다."""
    # 원본 이미지 복사
    vis_img = np.array(img).copy()
    
    # 마스크 오버레이 (반투명 색상)
    color_mask = np.zeros_like(vis_img)
    color_mask[mask > 0] = [0, 255, 0]  # 초록색 마스크
    
    # 마스크 적용 (알파 블렌딩)
    alpha = 0.4
    vis_img = cv2.addWeighted(vis_img, 1-alpha, color_mask, alpha, 0)
    
    return vis_img

def save_mask_data(mask, mask_id, save_dir, img_basename):
    """마스크 데이터를 저장합니다."""
    # 1. 마스크 이미지 저장 (0~255)
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_filename = f"{img_basename[:-4]}_mask_{mask_id:03d}.png"
    mask_path = os.path.join(save_dir, 'masks', mask_filename)
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    cv2.imwrite(mask_path, mask_uint8)
    
def main():
    checkpoint = "data/ckpt/sam2.1_hiera_large.pt"
    model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    print(f"SAM2 model loaded")
    # 이미지 로드
    cliff_data = joblib.load('examples_image/examples_image_cliff_hr48.joblib')
    img_names = cliff_data['imgname']
    bboxes = cliff_data['bboxes']
    
    img_dir = '/home/oem/members/dyub/CLIFF/examples_image/padded_imgs'
    img_path_list = [os.path.join(img_dir, img_path) for img_path in img_names]
    img_pil_batch = [np.array(Image.open(img_path).convert('RGB')) for img_path in img_path_list]
    boxes_batch = [bbox.astype(np.int32) for bbox in bboxes]
    print(f"Image and bbox batch loaded")
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image_batch(img_pil_batch)
        masks_batch, scores_batch, logits_batch = predictor.predict_batch(box_batch=boxes_batch)
        
        for i, (masks, scores) in enumerate(zip(masks_batch, scores_batch)):
            
            # 마스크 데이터 저장 (마스크 이미지 + JSON)
            img_basename = img_names[i]
            mask_id = 0
            for mask, score in zip(masks, scores):
                mask_id = mask_id + 1
                save_mask_data(mask, mask_id, img_dir, img_basename)
                
                # 시각화 이미지 생성 및 저장
                vis_img = create_mask_visualization(img_pil_batch[i], mask)
                vis_filename = f"{img_basename[:-4]}_visualization_{mask_id:03d}.png"
                vis_path = os.path.join(img_dir, 'seg_vis', vis_filename)
                os.makedirs(os.path.dirname(vis_path), exist_ok=True)
                
                vis_pil = Image.fromarray(vis_img.astype(np.uint8))
                vis_pil.save(vis_path)

            print(f"{img_basename} : {len(masks)} masks || mask and visualization images are saved")
        
        
if __name__ == "__main__":
    main()