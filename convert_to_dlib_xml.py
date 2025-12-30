import os
import random
from xml.dom.minidom import Document

# ================= 配置 =================
TXT_FILE = './processed_data/landmarks_list.txt'
TRAIN_XML = './processed_data/dlib_train.xml' # 训练用
TEST_XML  = './processed_data/dlib_test.xml'  # 测试用
IMG_SIZE = 224
SPLIT_RATIO = 0.9 # 90% 训练，10% 测试 (必须和 PyTorch 保持一致)
SEED = 42 # 固定随机种子，确保和 PyTorch 的切分逻辑尽量一致
# =======================================

def write_xml(lines, output_xml):
    doc = Document()
    dataset = doc.createElement('dataset')
    doc.appendChild(dataset)
    name_node = doc.createElement('name')
    name_node.appendChild(doc.createTextNode('300-W Split'))
    dataset.appendChild(name_node)
    images_node = doc.createElement('images')
    dataset.appendChild(images_node)
    
    for line in lines:
        parts = line.strip().split()
        filename = parts[0]
        coords = [float(x) for x in parts[1:]]
        
        img_node = doc.createElement('image')
        img_node.setAttribute('file', filename)
        box_node = doc.createElement('box')
        box_node.setAttribute('top', '0')
        box_node.setAttribute('left', '0')
        box_node.setAttribute('width', str(IMG_SIZE))
        box_node.setAttribute('height', str(IMG_SIZE))
        
        for i in range(68):
            part_node = doc.createElement('part')
            part_node.setAttribute('name', str(i).zfill(2))
            part_node.setAttribute('x', str(int(coords[2*i])))
            part_node.setAttribute('y', str(int(coords[2*i+1])))
            box_node.appendChild(part_node)
        img_node.appendChild(box_node)
        images_node.appendChild(img_node)
        
    with open(output_xml, 'w') as f:
        f.write(doc.toprettyxml(indent="  "))
    print(f"✅ 生成: {output_xml} (包含 {len(lines)} 张图片)")

def main():
    with open(TXT_FILE, 'r') as f:
        all_lines = f.readlines()
    
    # 打乱数据 (固定种子)
    random.seed(SEED)
    random.shuffle(all_lines)
    
    # 切分
    train_size = int(len(all_lines) * SPLIT_RATIO)
    train_lines = all_lines[:train_size]
    test_lines = all_lines[train_size:]
    
    # 分别写入
    write_xml(train_lines, TRAIN_XML)
    write_xml(test_lines, TEST_XML)

if __name__ == '__main__':
    main()