# ## **加载固化的模型进行预测**

import codecs
import os
import sys
import numpy as np
import time
import paddle
import paddle.fluid as fluid
import math
import functools
import json

from IPython.display import display
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from collections import namedtuple
from parameters import train_parameters, init_train_parameters

paddle.enable_static()
os.chdir(os.path.abspath(os.path.dirname(sys.argv[0])))
init_train_parameters()
ues_tiny = train_parameters['use_tiny']
yolo_config = train_parameters['yolo_tiny_cfg'] if ues_tiny else train_parameters['yolo_cfg']

target_size = yolo_config['input_size']
anchors = yolo_config['anchors']
anchor_mask = yolo_config['anchor_mask']
label_dict = train_parameters['num_dict']
class_dim = train_parameters['class_dim']
print("label_dict:{} class dim:{}".format(label_dict, class_dim))
place = fluid.CUDAPlace(0) if train_parameters['use_gpu'] else fluid.CPUPlace()
exe = fluid.Executor(place)
path = train_parameters['freeze_dir']
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe)


def draw_bbox_image(img, boxes, labels, save_name, scores, isFilter):
    """
    给图片画上外接矩形框
    :param img:
    :param boxes:
    :param save_name:
    :param labels
    :return:
    """
    customColors = ['red', 'yellow', 'blue', 'green', 'black', 'brown']

    font = ImageFont.truetype('./Arial.ttf', 100)
    font2 = ImageFont.truetype('./Arial.ttf', 60)
    img_width, img_height = img.size
    draw = ImageDraw.Draw(img)
    for box, label, score in zip(boxes, labels, scores):
        if isFilter:
            if score < 0.05:
                continue

        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        draw.rectangle((xmin, ymin, xmax, ymax),
                       None,
                       customColors[int(label)],
                       10)
        draw.text((xmin, ymin),
                  label_dict[int(label)],
                  font=font,
                  fill=(255, 255, 0))
        draw.text((xmin, ymax),
                  str(score),
                  font=font,
                  fill=(int(score*255), int(score*255), int(score*255)))
    img.save(save_name)
    #display(img)


def resize_img(img, target_size):
    """
    保持比例的缩放图片
    :param img:
    :param target_size:
    :return:
    """
    img = img.resize(target_size[1:], Image.BILINEAR)
    return img


def read_image(img_path):
    """
    读取图片
    :param img_path:
    :return:
    """
    origin = Image.open(img_path)
    img = resize_img(origin, target_size)
    resized_img = img.copy()
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
    img -= 127.5
    img *= 0.007843
    img = img[np.newaxis, :]
    return origin, img, resized_img

total_time = 0

def infer(image_path):
    origin, tensor_img, resized_img = read_image(image_path)
    input_w, input_h = origin.size[0], origin.size[1]
    image_shape = np.array([input_h, input_w], dtype='int32')
    # print("image shape high:{0}, width:{1}".format(input_h, input_w))
    t1 = time.time()
    batch_outputs = exe.run(inference_program,
                            feed={feed_target_names[0]: tensor_img,
                                  feed_target_names[1]: image_shape[np.newaxis, :]},
                            fetch_list=fetch_targets,
                            return_numpy=False)
    period = time.time() - t1
    print("predict cost time:{0}".format("%2.2f sec" % period))
    global total_time
    total_time += period
    bboxes = np.array(batch_outputs[0])
    # print(bboxes)

    # 用于展示一张图片用于预测的效果
    if bboxes.shape[1] != 6:
        print("No object found in {}".format(image_path))
        return
    labels = bboxes[:, 0].astype('int32').tolist()
    scores = bboxes[:, 1].astype('float32').tolist()
    boxes = bboxes[:, 2:].astype('float32').tolist()

    if image_path == "data/data6045/lslm-test/2.jpg":
        last_dot_index = image_path.rfind('.')
        out_path = image_path[:last_dot_index]
        out_path += '-result.jpg'
        draw_bbox_image(origin, boxes, labels, out_path)

    predict = []
    for i in range(len(labels)):
        predictTmp = []
        predictTmp.append(labels[i])
        predictTmp.append(scores[i])
        for j in boxes[i]:
            predictTmp.append(j)
        predict.append(predictTmp)
    tmp = image_path[25:-4]
    f = open("./input/detection-results/" + image_path.split('/')[-1].split('.')[0] + '.txt', 'w')
    for i in predict:
        for j in i:
            f.write(str(j) + ' ')
        f.write('\n')
    f.close()
    return predict


if __name__ == '__main__':
    if os.path.exists('./input') == False:
        os.mkdir('./input')
        os.mkdir('./input/detection-results')
        os.mkdir('./input/ground-truth')
    file_path = os.path.join(train_parameters['data_dir'], 'eval.txt')
    images = [line.strip() for line in open(file_path)]
    for line in images:
        image_path = line
        parts = line.split('\t')
        filename = parts[0]
        filename_path = os.path.join(train_parameters['data_dir'], parts[0])
        infer(filename_path)

        bbox_labels = []
        for object_str in parts[1:]:
            if len(object_str) <= 1:
                continue
            bbox_sample = []
            object = json.loads(object_str)
            bbox_sample.append(int(train_parameters['label_dict'][object['value']]))
            bbox = object['coordinate']
            bbox_sample.append(float(bbox[0][0]))
            bbox_sample.append(float(bbox[0][1]))
            bbox_sample.append(float(bbox[1][0]))
            bbox_sample.append(float(bbox[1][1]))
            bbox_labels.append(bbox_sample)
        f = open("./input/ground-truth/" + image_path.split('/')[-1].split('.')[0] + '.txt', 'w')
        for i in bbox_labels:
            for j in i:
                f.write(str(j) + ' ')
            f.write('\n')
        f.close()
        print("Infer total time:{0}".format("%2.2f sec" % total_time))
