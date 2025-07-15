import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# =================================================================================
# 1. 데이터셋 클래스 및 collate_fn 정의 (이전과 동일)
# =================================================================================
class CrackDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.images = sorted([f for f in os.listdir(root_dir) if f.endswith('.jpg')])
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.root_dir, self.images[idx])
            mask_path = os.path.splitext(img_path)[0] + '.png'
            image = cv2.imread(img_path)
            if image is None: return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                h, w, _ = image.shape
                mask = np.zeros((h, w), dtype=np.uint8)
            mask = (mask > 127).astype('float32')
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask'].unsqueeze(0)
            return image, mask
        except Exception:
            return None

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return None, None
    return torch.utils.data.dataloader.default_collate(batch)


# =================================================================================
# 2. 평가 함수 정의 (수정 버전)
# =================================================================================

def iou_score(output, target):
    with torch.no_grad():
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
        target = (target > 0.5).float()
        intersection = (output * target).sum()
        union = output.sum() + target.sum() - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()

def dice_score(output, target):
    with torch.no_grad():
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
        target = (target > 0.5).float()
        intersection = (output * target).sum()
        dice = (2. * intersection + 1e-6) / (output.sum() + target.sum() + 1e-6)
    return dice.item()

# [핵심 수정] evaluate 함수
def evaluate(model, loader, device):
    model.eval()
    total_iou = 0
    total_dice = 0
    num_samples = 0 # 샘플 단위로 정확히 계산하기 위해 변경
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            # collate_fn이 None을 반환하는 경우 건너뛰기
            if images is None:
                continue

            images = images.to(device)
            masks = masks.to(device)
            
            # --- [수정] 마지막 배치 차원 문제 해결 ---
            # 만약 배치 크기가 1이어서 배치 차원이 사라졌다면 (3D 텐서),
            # 수동으로 배치 차원(4D)을 추가해 줍니다.
            if images.dim() == 3:
                images = images.unsqueeze(0)
            
            preds = model(images)

            # 배치 내 각 샘플에 대해 점수를 계산하고 합산
            for i in range(preds.shape[0]):
                total_iou += iou_score(preds[i], masks[i])
                total_dice += dice_score(preds[i], masks[i])
            
            num_samples += images.shape[0]

    if num_samples > 0:
        avg_iou = total_iou / num_samples
        avg_dice = total_dice / num_samples
    else:
        avg_iou, avg_dice = 0, 0
    
    print("\n" + "="*30 + "\n      Model Performance\n" + "="*30)
    print(f"  - Evaluated Samples   : {num_samples}")
    print(f"  - Average IoU (mIoU)  : {avg_iou:.4f}")
    print(f"  - Average Dice Score  : {avg_dice:.4f}")
    print("="*30)


# =================================================================================
# 3. 메인 실행부
# =================================================================================

if __name__ == '__main__':
    # --- 설정 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = r"C:\Users\dromii\Downloads\unetpp_crack.pth"
    EVAL_DATA_PATH = r"D:\datasets\UAV_crack_pothole_datasets\pavement crack datasets-20210103T153625Z-001\pavement crack datasets\CRACK500\traincrop\traincrop"
    BATCH_SIZE = 2
    IMAGE_HEIGHT = 384
    IMAGE_WIDTH = 640

    print(f"Using device: {DEVICE}")
    print(f"Loading model from: {MODEL_PATH}")
    print(f"Evaluating on dataset: {EVAL_DATA_PATH}")

    # --- 모델 로드 ---
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}"); exit()
    model.eval()

    # --- 데이터 로더 설정 (drop_last=False 유지) ---
    eval_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])
    eval_dataset = CrackDataset(root_dir=EVAL_DATA_PATH, transform=eval_transform)
    
    # [수정] drop_last를 사용하지 않고, collate_fn으로 손상된 데이터만 처리
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # --- 평가 실행 ---
    evaluate(model, eval_loader, DEVICE)