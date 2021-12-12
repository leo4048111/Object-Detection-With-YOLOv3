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
#path = train_parameters['freeze_dir']
path = 'freeze_model_2'
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
            if score < 0.15:
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


def infer(image_path, output_path, openImg, isFilter):
    """
    预测，将结果保存到一副新的图片中
    :param image_path:
    :return:
    """
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

    bboxes = np.array(batch_outputs[0])
    # print(bboxes)

    if bboxes.shape[1] != 6:
        print("No object found in {}".format(image_path))
        return
    labels = bboxes[:, 0].astype('int32')
    scores = bboxes[:, 1].astype('float32')
    boxes = bboxes[:, 2:].astype('float32')

    out_path = output_path
    out_path += '/' + os.path.splitext(os.path.basename(image_path))[0]+'-result.jpg'
    draw_bbox_image(origin, boxes, labels, out_path, scores, isFilter)
    openImg(out_path)


if __name__ == '__main__':
    '''image_name = sys.argv[1]
    image_path = image_name'''
    #test as main
    image_path = "data/data116055/pbcTest/10.jpg"
    output_path = os.path.getcwd()
    infer(image_path, output_path, None)
