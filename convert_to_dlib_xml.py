import os
import cv2
from xml.dom.minidom import Document

# ================= 配置 =================
# 必须和 PyTorch 训练用的数据路径一致
IMG_DIR = './processed_data'
TXT_FILE = './processed_data/landmarks_list.txt'
XML_SAVE_PATH = './processed_data/training_with_face_landmarks.xml'
IMG_SIZE = 224 
# =======================================

def create_dlib_xml(img_dir, txt_file, output_xml):
    doc = Document()
    dataset = doc.createElement('dataset')
    doc.appendChild(dataset)
    
    name_node = doc.createElement('name')
    name_node.appendChild(doc.createTextNode('300-W Processed'))
    dataset.appendChild(name_node)
    
    images_node = doc.createElement('images')
    dataset.appendChild(images_node)
    
    print("正在生成 XML 文件...")
    
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        filename = parts[0]
        # 还原坐标：txt里存的是绝对坐标 (0~224)，直接用即可
        coords = [float(x) for x in parts[1:]]
        
        # 创建 image 节点
        # 注意：Dlib XML 里的 file 路径是相对于 xml 文件的
        img_node = doc.createElement('image')
        img_node.setAttribute('file', filename)
        
        # 创建 box 节点
        # 因为我们的图片已经是裁剪好的人脸 (224x224)，
        # 所以 bounding box 就是整张图的大小
        box_node = doc.createElement('box')
        box_node.setAttribute('top', '0')
        box_node.setAttribute('left', '0')
        box_node.setAttribute('width', str(IMG_SIZE))
        box_node.setAttribute('height', str(IMG_SIZE))
        
        # 添加 68 个关键点
        for i in range(68):
            x = coords[2 * i]
            y = coords[2 * i + 1]
            
            part_node = doc.createElement('part')
            part_node.setAttribute('name', str(i).zfill(2)) # 00, 01, ...
            part_node.setAttribute('x', str(int(x)))
            part_node.setAttribute('y', str(int(y)))
            box_node.appendChild(part_node)
            
        img_node.appendChild(box_node)
        images_node.appendChild(img_node)
        
    with open(output_xml, 'w') as f:
        f.write(doc.toprettyxml(indent="  "))
        
    print(f"✅ XML 已生成: {output_xml}")

if __name__ == '__main__':
    create_dlib_xml(IMG_DIR, TXT_FILE, XML_SAVE_PATH)