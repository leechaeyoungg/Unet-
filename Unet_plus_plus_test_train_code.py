import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


#데이터셋 정의
class CrackDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.images = sorted([f for f in os.listdir(root_dir) if f.endswith('.jpg')])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        mask_path = img_path.replace('.jpg', '.png')

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype('float32')

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)

        return image, mask


def main():
    # 학습 설정
    root_path = r"D:\datasets\UAV_crack_pothole_datasets\pavement crack datasets-20210103T153625Z-001\pavement crack datasets\CRACK500\traincrop\traincrop"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 전처리 및 증강
    transform = A.Compose([
        A.Resize(384, 640), #사이즈 모델 특성상 32로 나누어 떨어져야해서 원본 이미지사이즈 (360,640) 에서 변경 
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    # 데이터로더
    train_dataset = CrackDataset(root_dir=root_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

    # 모델 정의 (Unet++)
    model = smp.UnetPlusPlus(
        encoder_name="resnet34", #Unet++의 인코더(backbone)로 사용할 모델 지정
        encoder_weights="imagenet", #resnet34를 ImageNet 사전 학습된 가중치로 초기화
        in_channels=3, #입력 이미지의 채널 수 (R,G,B)
        classes=1 #출력할 마스크의 채널 수 (각 픽셀이 크랙일 확률을 예측하는 1개의 채널만 출력)
    ).to(device)

    # 손실 함수 및 최적화기
    loss_fn = smp.losses.DiceLoss(mode='binary')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 학습 루프
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)

            preds = model(images)
            loss = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(train_loader):.4f}")

    # 모델 저장
    torch.save(model.state_dict(), "unetpp_crack.pth")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
