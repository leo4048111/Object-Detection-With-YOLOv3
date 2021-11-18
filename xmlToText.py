"""
需要修改的地方 ：
1、你自己的类别 CLASSES
2、数据集路劲 data_path
3、第12行，生成文件的保存路径及名称
"""
import xml.etree.ElementTree as ET
import os

CLASSES = ['person', 'bird', 'cat', 'cow'
           ,'dog','horse','sheep','aeroplane','bicycle'
           ,'boat','bus','car','motorbike','train'
           ,'bottle','chair','dining table','potted plant'
           ,'sofa' , 'tv/monitor']

def convert_xml_annotation(data_path, classes):
    xml_dir = []
    for xml in os.listdir(data_path):
        if xml.endswith('.xml'):
            xml_dir.append(xml)
    print("Total xml files : ", len(xml_dir))
    with open("D:/Project/PY/PaddlePadlleYoloV3/train.txt", 'w') as f:
        for i in range(len(xml_dir)):
            tree = ET.parse(data_path + xml_dir[i])
            root = tree.getroot()

            # image path
            filename = root.find('filename').text
            image_path = data_path + filename
            ##annotation = image_path
            annotation = filename

            # coordinates of label : xmin  ymin  xmax  ymax
            for obj in root.iter('object'):
                ##difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes:
                    continue
                cls_id = classes.index(cls)
                bbox = obj.find('bndbox')
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                ##annotation += ' ' + ','.join([xmin, ymin, xmax, ymax,str(cls_id)])
                annotation += ' ' + '{"value":"' +cls+ '","coordinate":[[' + xmin+',' + ymin +'],[' + xmax +',' + ymax + ']]}'
            print(annotation)
            f.write(annotation + "\n")


convert_xml_annotation("D:/Project/PY/PaddlePadlleYoloV3/Annotations/", CLASSES)
