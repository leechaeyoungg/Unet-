import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, jaccard_score
import segmentation_models_pytorch as smp

# 1. 모델 로드
def load_model(path, device):
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval().to(device)
    return model

# 2. 평가 및 시각화 함수
def evaluate_and_visualize(model, image_path, mask_path, device):
    # 이미지 불러오기 및 전처리
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (640, 384))
    img_tensor = torch.tensor(img_resized / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(device)

    # 마스크 불러오기
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_resized = cv2.resize(mask, (640, 384))
    mask_bin = (mask_resized > 127).astype(np.uint8)

    # 예측
    with torch.no_grad():
        pred = model(img_tensor)
        pred_bin = (pred.sigmoid().cpu().numpy()[0, 0] > 0.3).astype(np.uint8)

    # 평가 지표
    iou = jaccard_score(mask_bin.flatten(), pred_bin.flatten(), zero_division=1)
    precision = precision_score(mask_bin.flatten(), pred_bin.flatten(), zero_division=1)
    recall = recall_score(mask_bin.flatten(), pred_bin.flatten(), zero_division=1)

    # 시각화
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(img_resized); plt.title("Input Image"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(mask_bin, cmap='gray'); plt.title("Ground Truth"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(pred_bin, cmap='gray'); plt.title("Prediction"); plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 지표 출력
    print(f"IoU: {iou:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

# 3. 실행 예시
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = r"C:\Users\dromii\Downloads\unetpp_crack.pth"
image_path = r"D:\datasets\UAV_crack_pothole_datasets\pavement crack datasets-20210103T153625Z-001\pavement crack datasets\CRACK500\traincrop\traincrop\20160222_081011_1_361.jpg"
mask_path = r"D:\datasets\UAV_crack_pothole_datasets\pavement crack datasets-20210103T153625Z-001\pavement crack datasets\CRACK500\traincrop\traincrop\20160222_081011_1_361.png"

model = load_model(model_path, device)
evaluate_and_visualize(model, image_path, mask_path, device)
