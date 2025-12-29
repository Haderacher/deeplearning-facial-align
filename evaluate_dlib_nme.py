import dlib
import cv2
import numpy as np
import os
from tqdm import tqdm

# ================= 配置 =================
MODEL_PATH = 'dlib_baseline_model.dat'
IMG_DIR = './processed_data'
LABEL_FILE = './processed_data/landmarks_list.txt'
IMG_SIZE = 224
# =======================================

def calculate_nme(pred_points, gt_points):
    """计算单张图片的 NME (使用双眼外眼角距离归一化)"""
    pred_points = np.array(pred_points)
    gt_points = np.array(gt_points)
    
    # 36: 左眼角, 45: 右眼角 (基于 68 点定义)
    left_eye = gt_points[36]
    right_eye = gt_points[45]
    
    # 计算归一化距离 d (Inter-ocular distance)
    d = np.linalg.norm(left_eye - right_eye)
    if d < 1e-6: d = 1.0
    
    # 计算所有点的平均欧氏距离
    error = np.mean(np.linalg.norm(pred_points - gt_points, axis=1))
    return error / d

def evaluate():
    # 1. 加载模型
    print("正在加载 Dlib 模型...")
    detector = dlib.get_frontal_face_detector() # 虽然不需要检测，但predictor依赖box
    predictor = dlib.shape_predictor(MODEL_PATH)
    
    # 2. 读取标签列表
    with open(LABEL_FILE, 'r') as f:
        lines = f.readlines()
    
    # 为了公平，这里应该用和 PyTorch 一样的验证集划分
    # 简单起见，我们假设最后 10% 是验证集 (与 train.py 保持一致)
    val_size = int(0.1 * len(lines))
    val_lines = lines[-val_size:] 
    
    print(f"开始在验证集上评估 ({len(val_lines)} 张图片)...")
    
    total_nme = 0.0
    count = 0
    
    for line in tqdm(val_lines):
        parts = line.strip().split()
        img_name = parts[0]
        gt_coords = [float(x) for x in parts[1:]]
        gt_points = np.array(gt_coords).reshape(68, 2)
        
        img_path = os.path.join(IMG_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        
        # 转为灰度图 (Dlib 推荐)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 3. 构造 Bounding Box
        # 因为我们是在 crop 好的图上跑，所以 Box 就是整张图
        h, w = gray.shape
        rect = dlib.rectangle(0, 0, w, h)
        
        # 4. 预测
        shape = predictor(gray, rect)
        
        # 提取预测坐标
        pred_points = []
        for i in range(68):
            pred_points.append([shape.part(i).x, shape.part(i).y])
            
        # 5. 计算 NME
        nme = calculate_nme(pred_points, gt_points)
        total_nme += nme
        count += 1
        
    avg_nme = total_nme / count
    print("\n" + "="*40)
    print(f"Dlib Baseline 模型评估结果:")
    print(f"测试样本数: {count}")
    print(f"平均 NME: {avg_nme:.4f}")
    print("="*40)

if __name__ == '__main__':
    if os.path.exists(MODEL_PATH):
        evaluate()
    else:
        print("错误：找不到模型文件，请先运行 train_dlib.py")