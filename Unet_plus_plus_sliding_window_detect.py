import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ----------------------------
# 모델 로드 함수
# ----------------------------
def load_model(model_path, device):
    model = smp.UnetPlusPlus(
        encoder_name="resnet34", #백본 모델
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    return model

# ----------------------------
# 슬라이딩 윈도우 기반 예측
# ----------------------------
def sliding_window_prediction(image, model, window_size=(384, 640), stride=(256, 448), device='cuda'):
    #window_size: 모델이 입력으로 받는 크기, stride: 윈도우 간 이동 간격 (겹침 가능)
    h, w, _ = image.shape
    out_mask = np.zeros((h, w), dtype=np.float32)
    count_mask = np.zeros((h, w), dtype=np.float32) #겹치는 부분 평균 계산용 가중치 (결과 부드럽게)

    transform = A.Compose([
        A.Resize(*window_size),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    for y in tqdm(range(0, h - window_size[0] + 1, stride[0])):
        for x in range(0, w - window_size[1] + 1, stride[1]):
            crop = image[y:y + window_size[0], x:x + window_size[1]]
            augmented = transform(image=crop)
            input_tensor = augmented['image'].unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.sigmoid(output).cpu().numpy()[0, 0]  # shape: (384, 640)

            out_mask[y:y + window_size[0], x:x + window_size[1]] += pred
            count_mask[y:y + window_size[0], x:x + window_size[1]] += 1

    # Normalize overlapped regions
    out_mask = out_mask / np.maximum(count_mask, 1e-5)
    return (out_mask > 0.5).astype(np.uint8), out_mask  # (binary_mask, raw_score)

def visualize_overlay(image, binary_mask, save_path="D:/crack_result_overlay.png"):
    color_mask = np.zeros_like(image)
    color_mask[:, :, 0] = binary_mask * 255  # Red only

    # 배경 밝기 보존 + 마스크 부분만 덧입히기
    mask_area = binary_mask.astype(bool)
    overlay = image.copy()
    overlay[mask_area] = cv2.addWeighted(image[mask_area], 0.5, color_mask[mask_area], 0.5, 0)

    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"예측 오버레이 저장 완료: {save_path}")




# ----------------------------
# 실행부
# ----------------------------
if __name__ == '__main__':
    image_path = r"D:\crack.jpg"
    model_path = r"C:\Users\dromii\Downloads\unetpp_crack.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    model = load_model(model_path, device)
    binary_mask, score_map = sliding_window_prediction(image, model, device=device)

    #시각화
    visualize_overlay(image, binary_mask, save_path="D:/crack_result_overlay1.png")

