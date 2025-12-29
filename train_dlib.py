import dlib
import os
import time

# ================= 配置 =================
XML_FILE = './processed_data/training_with_face_landmarks.xml'
OUTPUT_MODEL = 'dlib_baseline_model.dat'
# =======================================

def train_dlib():
    # 1. 设置训练参数
    options = dlib.shape_predictor_training_options()
    
    # --- 关键超参数 (可以写进实验报告) ---
    options.oversampling_amount = 10  # 数据增强倍数 (随机变形)
    options.nu = 0.1                  # 正则化参数 (0.1 是标准值)
    options.tree_depth = 4            # 回归树的深度 (通常 4-5)
    options.cascade_depth = 12        # 级联层数 (层数越多越准，但模型越大)
    options.feature_pool_size = 400   # 每次分裂节点时随机采样的像素点数
    options.num_test_splits = 50      # 节点分裂尝试次数
    
    # 打印训练日志
    options.be_verbose = True
    
    # 开启多核训练
    options.num_threads = 4 

    print(f"开始训练 Dlib 模型 (ERT算法)...")
    print(f"输入数据: {XML_FILE}")
    
    start_time = time.time()
    
    # 2. 开始训练
    # 注意：Dlib 可能会把一部分数据划分为验证集，但我们为了最大化训练数据，
    # 这里直接用全部数据训练，后面单独手写验证代码
    dlib.train_shape_predictor(XML_FILE, OUTPUT_MODEL, options)
    
    end_time = time.time()
    print(f"✅ 训练完成！耗时: {end_time - start_time:.2f} 秒")
    print(f"模型已保存为: {OUTPUT_MODEL}")

if __name__ == '__main__':
    # 确保 XML 存在
    if not os.path.exists(XML_FILE):
        print("错误：找不到 XML 文件，请先运行 convert_to_dlib_xml.py")
    else:
        train_dlib()