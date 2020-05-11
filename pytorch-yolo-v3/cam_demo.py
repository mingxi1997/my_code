from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random
import argparse
import pickle as pkl
from PIL import Image, ImageFont, ImageDraw


class Yolov3Manager(object):
    def __init__(self,callback):
        self.Callback=callback
        self.__start=True
# import onnx
# from onnx_tf.backend import prepare

# def get_test_input(input_dim, CUDA):
#     img = cv2.imread("imgs/messi.jpg")
#     img = cv2.resize(img, (input_dim, input_dim))
#     img_ = img[:, :, ::-1].transpose((2, 0, 1))
#     img_ = img_[np.newaxis, :, :, :] / 255.0
#     img_ = torch.from_numpy(img_).float()
#     img_ = Variable(img_)
#
#     if CUDA:
#         img_ = img_.cuda()
#
#     return img_


    def prep_image(self, img, inp_dim):
        """
        Prepare image for inputting to the neural network.

        Returns a Variable
        """

        orig_im = img
        dim = orig_im.shape[1], orig_im.shape[0]
        img = cv2.resize(orig_im, (inp_dim, inp_dim))
        img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
        img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
        return img_, orig_im, dim


    def write(self, x, img):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        cls = int(x[-1])
        label = "{0}".format(classes[cls])
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
        return img


    def arg_parse(self):
        """
        Parse arguements to the detect module

        """

        parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
        parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.25)
        parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
        parser.add_argument("--reso", dest='reso', help=
        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                            default="160", type=str)
        return parser.parse_args()


    # def onnx2pb(onnx_input_path, pb_output_path):
    #     onnx_model = onnx.load(onnx_input_path)  # load onnx model
    #     tf_exp = prepare(onnx_model)  # prepare tf representation
    #     tf_exp.export_graph(pb_output_path)  # export the model
    def stop(self):
        self.__start=False

    def start(self):
        self.__start=True
        cfgfile = "cfg/yolov3.cfg"
        weightsfile = "yolov3.weights"
        num_classes = 80
        font = ImageFont.truetype(font='font/simsun.ttc', size=16)
        args = self.arg_parse()
        confidence = float(args.confidence)
        nms_thesh = float(args.nms_thresh)
        start = 0
        CUDA = torch.cuda.is_available()

        mean_num = {
            'bottle': [0] * 8,
            'person': [0] * 8,
            'cup': [0] * 8,
        }

        num_classes = 80
        bbox_attrs = 5 + num_classes

        model = Darknet(cfgfile)
        model.load_weights(weightsfile)

        # weights -> onx
        # x = torch.randn(1, 3, 320, 320)
        # save_onnx_name = 'yolov3.onnx'
        # torch.onnx.export(model,
        #                   x,
        #                   save_onnx_name,
        #                   opset_version=10,
        #                   do_constant_folding=True,  # 是否执行常量折叠优化
        #                   input_names=["input"],  # 输入名
        #                   output_names=["output"],  # 输出名
        #                   dynamic_axes={"input": {0: "batch_size"},  # 批处理变量
        #                                 "output": {0: "batch_size"}})
        # print('over')
        # exit()


        model.net_info["height"] = args.reso
        inp_dim = int(model.net_info["height"])

        assert inp_dim % 32 == 0
        assert inp_dim > 32

        if CUDA:
            model.cuda()

        model.eval()

        videofile = 'video.avi'

        cap = cv2.VideoCapture(0)

        assert cap.isOpened(), 'Cannot capture source'

        frames = 0
        start = time.time()
        person = 0
        cup = 0
        bottle = 0
        while cap.isOpened() or self.__start:

            ret, frame = cap.read()
            if ret:
                PIL_img = Image.fromarray(frame[:, :, ::-1])
                draw = ImageDraw.Draw(PIL_img)
                # print(frame.shape,'-----------------')
                img, orig_im, dim = self.prep_image(frame, inp_dim)

                #            im_dim = torch.FloatTensor(dim).repeat(1,2)

                if CUDA:
                    im_dim = im_dim.cuda()
                    img = img.cuda()
                output = model(Variable(img))
                output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

                # if type(output) == int:
                #     frames += 1
                #     print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
                #     cv2.imshow("frame", orig_im)
                #     key = cv2.waitKey(1)
                #     if key & 0xFF == ord('q'):
                #         break
                #     continue

                output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim)) / inp_dim

                #            im_dim = im_dim.repeat(output.size(0), 1)
                output[:, [1, 3]] *= frame.shape[1]
                output[:, [2, 4]] *= frame.shape[0]

                classes = load_classes('data/coco.names')
                colors = pkl.load(open("pallete", "rb"))
                num_objects = dict.fromkeys(classes, 0)
                for i in output:
                    if classes[int(i[-1])] in ['person', 'bottle', 'cup']:
                        num_objects[classes[int(i[-1])]] += 1
                mean_num['bottle'].append(num_objects['bottle'])
                mean_num['bottle'].pop(0)
                mean_num['cup'].append(num_objects['cup'])
                mean_num['cup'].pop(0)
                mean_num['person'].append(num_objects['person'])
                mean_num['person'].pop(0)
                # print(mean_num)
                cup_now = max(mean_num['cup'], key=mean_num['cup'].count)
                person_now = max(mean_num['person'], key=mean_num['person'].count)
                bottle_now = max(mean_num['bottle'], key=mean_num['bottle'].count)
                # if cup != cup_now:
                #     # 输出 杯子个数 cup_now 个
                #     cup = cup_now
                #     pass
                # if person_now != person:
                #     # 输出 人个数 person_now 个
                #     person = person_now
                #     pass
                # if bottle_now != bottle:
                #     # 输出 瓶子个数 bottle_now 个
                #     bottle = bottle_now
                #     pass

                # draw.text((1, 20), '杯子: ' + str(cup_now), fill='red', font=font)
                # draw.text((1, 40), '瓶子: ' + str(bottle_now), fill='red', font=font)
                # draw.text((1, 60), '人: ' + str(person_now), fill='red', font=font)
                cv_img = np.array(PIL_img)[..., ::-1]
                # cv2.imshow('result', cv_img)
                # list(map(lambda x: write(x, orig_im), output))
                # cv2.imshow("frame", orig_im)
                # key = cv2.waitKey(1)
                # if key & 0xFF == ord('q'):
                #     break
                frames += 1
                fps = str(frames / (time.time() - start))
                if self.__start==False:
                    break
                # print(str(bottle_now))
                if self.Callback:
                    self.Callback(str(bottle_now),str(cup_now),str(person_now),cv_img,fps)

                # print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))


            else:
                break


if __name__ == "__main__":
    yolov3=Yolov3Manager(None)
    yolov3.start()