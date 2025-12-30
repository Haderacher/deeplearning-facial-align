import dlib
import cv2
import numpy as np
import os
from tqdm import tqdm
from xml.dom.minidom import parse

# ================= 配置参数 =================
# 1. Dlib 训练好的模型路径
MODEL_PATH = 'dlib_baseline_model.dat'

# 2. 测试集 XML 路径 (由 convert_split_xml.py 生成)
TEST_XML_PATH = './processed_data/dlib_test.xml'

# 3. 图片所在的根目录
IMG_DIR = './processed_data' 
# ===========================================

def calculate_nme(pred_points, gt_points):
    """
    计算单张图片的 NME (归一化平均误差)
    归一化因子: 双眼外眼角间距 (Inter-ocular distance)
    """
    pred_points = np.array(pred_points)
    gt_points = np.array(gt_points)
    
    # 36: 左眼角, 45: 右眼角 (基于 68 点定义)
    left_eye = gt_points[36]
    right_eye = gt_points[45]
    
    # 计算归一化距离 d
    d = np.linalg.norm(left_eye - right_eye)
    
    # 防止除以 0 (虽然极不可能)
    if d < 1e-6: d = 1.0
    
    # 计算所有关键点的平均欧氏距离
    error = np.mean(np.linalg.norm(pred_points - gt_points, axis=1))
    return error / d

def get_test_data_from_xml(xml_path):
    """
    解析 XML 文件，提取图片文件名和真实坐标(GT)
    """
    print(f"正在解析测试集 XML: {xml_path} ...")
    dom = parse(xml_path)
    root = dom.documentElement
    images = root.getElementsByTagName('image')
    
    test_data = []
    
    for img_node in images:
        filename = img_node.getAttribute('file')
        
        # 提取 GT 坐标
        gt_points = np.zeros((68, 2), dtype=np.float32)
        box = img_node.getElementsByTagName('box')[0]
        parts = box.getElementsByTagName('part')
        
        # Dlib XML 的 part name 通常是 '00', '01'... 需要转为 int 索引
        for part in parts:
            idx = int(part.getAttribute('name'))
            x = float(part.getAttribute('x'))
            y = float(part.getAttribute('y'))
            gt_points[idx] = [x, y]
            
        test_data.append({
            'filename': filename,
            'gt_points': gt_points
        })
        
    return test_data

def evaluate():
    # 1. 检查文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 {MODEL_PATH}")
        return
    if not os.path.exists(TEST_XML_PATH):
        print(f"错误: 找不到测试集 XML {TEST_XML_PATH}")
        print("请先运行 convert_split_xml.py 生成数据集。")
        return

    # 2. 加载模型
    print("正在加载 Dlib 模型...")
    predictor = dlib.shape_predictor(MODEL_PATH)
    
    # 3. 获取测试数据
    test_data = get_test_data_from_xml(TEST_XML_PATH)
    print(f"测试集包含样本数: {len(test_data)}")
    
    total_nme = 0.0
    count = 0
    
    # 4. 循环评估
    print("开始评估...")
    for item in tqdm(test_data):
        # 拼接图片路径
        # XML 中的 filename 可能是相对路径，也可能只是文件名
        # 我们假设它相对于 IMG_DIR
        img_path = os.path.join(IMG_DIR, os.path.basename(item['filename']))
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图片 {img_path}")
            continue
            
        # 转灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 构造包围盒 (因为图片已经是 Crop 好的脸，所以 Box = 全图)
        rect = dlib.rectangle(0, 0, w, h)
        
        # 预测
        shape = predictor(gray, rect)
        
        # 提取预测坐标
        pred_points = []
        for i in range(68):
            pred_points.append([shape.part(i).x, shape.part(i).y])
            
        # 计算 NME
        nme = calculate_nme(pred_points, item['gt_points'])
        total_nme += nme
        count += 1
        
    # 5. 输出最终结果
    if count > 0:
        avg_nme = total_nme / count
        print("\n" + "="*50)
        print(f"Dlib 模型最终评估结果 (基于独立测试集):")
        print(f"测试样本数量: {count}")
        print(f"平均 NME     : {avg_nme:.5f}")
        print("="*50)
    else:
        print("未处理任何有效图片。")

if __name__ == '__main__':
    evaluate()