# calculate_nme.py
import torch
import numpy as np
from train import get_model, FaceLandmarksDataset # 导入你的模型和数据集类
from torch.utils.data import DataLoader

def calculate_nme(model_path, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    total_nme = 0.0
    count = 0
    
    with torch.no_grad():
        for images, landmarks in dataloader:
            images = images.to(device)
            # landmarks: (B, 136) normalized
            
            preds = model(images).cpu().numpy().reshape(-1, 68, 2)
            gts = landmarks.numpy().reshape(-1, 68, 2)
            
            # 计算 NME
            for i in range(len(preds)):
                pred = preds[i]
                gt = gts[i]
                
                # 计算双眼外眼角距离作为归一化因子 d (下标根据 68点定义: 36左眼角, 45右眼角)
                # 注意：Python索引从0开始，所以是 36 和 45
                left_eye = gt[36]
                right_eye = gt[45]
                d = np.linalg.norm(left_eye - right_eye)
                
                # 如果 d 太小(防止除0)，可以用 bounding box 对角线，这里简化处理
                if d < 1e-6: d = 1.0 
                
                # 计算 L2 距离均值
                error = np.mean(np.linalg.norm(pred - gt, axis=1))
                total_nme += (error / d)
                count += 1
                
    return total_nme / count

# 主程序
dataset = FaceLandmarksDataset('./processed_data', './processed_data/landmarks_list.txt')
# 这里应该用测试集，简单起见用全集或切分出的验证集
loader = DataLoader(dataset, batch_size=32, shuffle=False)

nme_mse = calculate_nme('best_model_mse.pth', loader)
nme_wing = calculate_nme('best_model_wing.pth', loader)

print(f"MSE Model NME:  {nme_mse:.4f} (越低越好)")
print(f"Wing Model NME: {nme_wing:.4f} (越低越好)")
print(f"性能提升: {((nme_mse - nme_wing)/nme_mse)*100:.2f}%")