import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train import get_model, FaceLandmarksDataset # 从你的训练代码导入

# ================= 配置 =================
MSE_MODEL_PATH = 'best_model_mse.pth'
WING_MODEL_PATH = 'best_model_wing.pth'
DATA_DIR = './processed_data'
LABEL_FILE = './processed_data/landmarks_list.txt'
IMG_SIZE = 224
TOP_K = 3 # 展示差距最大的前3张图
# =======================================

def calculate_sample_nme(pred, gt):
    """计算单张图片的 NME"""
    pred = pred.reshape(-1, 2)
    gt = gt.reshape(-1, 2)
    # 使用双眼距离归一化
    left_eye = gt[36]
    right_eye = gt[45]
    d = np.linalg.norm(left_eye - right_eye)
    if d < 1e-6: d = 1.0
    return np.mean(np.linalg.norm(pred - gt, axis=1)) / d

def visualize_comparison():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载两个模型
    print("正在加载模型...")
    model_mse = get_model().to(device)
    model_mse.load_state_dict(torch.load(MSE_MODEL_PATH, map_location=device))
    model_mse.eval()
    
    model_wing = get_model().to(device)
    model_wing.load_state_dict(torch.load(WING_MODEL_PATH, map_location=device))
    model_wing.eval()
    
    # 2. 加载数据
    # 这里建议使用验证集或者测试集，为了方便演示我们遍历整个数据集
    dataset = FaceLandmarksDataset(DATA_DIR, LABEL_FILE)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    results = []
    
    print("正在寻找最佳对比样本（这可能需要一点时间）...")
    with torch.no_grad():
        for i, (image, landmarks) in enumerate(loader):
            image = image.to(device)
            landmarks_gt = landmarks.numpy()[0] # (136,)
            
            # 预测
            pred_mse = model_mse(image).cpu().numpy()[0]
            pred_wing = model_wing(image).cpu().numpy()[0]
            
            # 计算各自的 NME
            nme_mse = calculate_sample_nme(pred_mse, landmarks_gt)
            nme_wing = calculate_sample_nme(pred_wing, landmarks_gt)
            
            # 记录差异 (MSE误差 - Wing误差)，值越大说明 Wing 提升越明显
            diff = nme_mse - nme_wing
            
            # 保存原图以便画图
            img_np = image.cpu().numpy()[0].transpose(1, 2, 0) # CHW -> HWC
            
            results.append({
                'img': img_np,
                'pred_mse': pred_mse,
                'pred_wing': pred_wing,
                'gt': landmarks_gt,
                'nme_mse': nme_mse,
                'nme_wing': nme_wing,
                'diff': diff
            })
            
            if i > 500: break # 稍微限制一下数量，不然跑太久

    # 3. 按差异从大到小排序，取前 K 个
    results.sort(key=lambda x: x['diff'], reverse=True)
    top_results = results[:TOP_K]
    
    print(f"找到差异最大的前 {TOP_K} 张图片，准备绘图...")

    # 4. 绘图
    for idx, item in enumerate(top_results):
        img = item['img']
        # 反归一化到像素坐标
        h, w = img.shape[:2] # 应该是 224, 224
        
        # 还原图片像素值 [0,1] -> [0,255]
        img_disp = (img * 255).astype(np.uint8)
        img_disp = cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR) # Matplotlib需要RGB，CV2画点需要BGR逻辑，这里有点绕，统一用RGB画图
        img_rgb = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # --- 左图：MSE ---
        ax1 = axes[0]
        img_mse = img_rgb.copy()
        pts_mse = item['pred_mse'].reshape(-1, 2) * IMG_SIZE
        gt_pts = item['gt'].reshape(-1, 2) * IMG_SIZE
        
        # 画点
        for (x, y) in pts_mse:
            cv2.circle(img_mse, (int(x), int(y)), 2, (255, 0, 0), -1) # Red for Prediction
        # 可选：画 GT (绿色) 做参考
        # for (x, y) in gt_pts:
        #     cv2.circle(img_mse, (int(x), int(y)), 1, (0, 255, 0), -1) 

        ax1.imshow(img_mse)
        ax1.set_title(f"MSE Prediction\nNME: {item['nme_mse']:.4f}", color='red', fontweight='bold')
        ax1.axis('off')
        
        # --- 右图：Wing ---
        ax2 = axes[1]
        img_wing = img_rgb.copy()
        pts_wing = item['pred_wing'].reshape(-1, 2) * IMG_SIZE
        
        for (x, y) in pts_wing:
            cv2.circle(img_wing, (int(x), int(y)), 2, (0, 255, 0), -1) # Green for Wing Prediction
            
        ax2.imshow(img_wing)
        ax2.set_title(f"Wing Prediction (Ours)\nNME: {item['nme_wing']:.4f}", color='green', fontweight='bold')
        ax2.axis('off')
        
        plt.suptitle(f"Sample #{idx+1} Comparison (Difference: {item['diff']:.4f})", fontsize=14)
        plt.tight_layout()
        save_path = f'comparison_sample_{idx+1}.png'
        plt.savefig(save_path, dpi=300)
        print(f"已保存对比图: {save_path}")
        plt.show()

if __name__ == "__main__":
    visualize_comparison()