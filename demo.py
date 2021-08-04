#!/usr/bin/python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys
import cv2
import jetson.inference
import jetson.utils

import argparse

from segnet_utils import *

from mot_opencv import MOTwithOpencv
from lane_detection_opencv import RegionLaneDetectionwithOpencv

import numpy as np



colors=[(51,255,255),(51,255,51),(255,102,178)]

target_list       = [1, 2, 3, 4, 6, 7, 10, 12, 13]
human_list        = [1, 2]
car_list          = [3, 4, 6, 7]
traffic_sign_list = [10, 12, 13]

def bbox_3D(det, image):
    ref_h = int(image.shape[0]*0.5+0.5)
    ref_w = int(image.shape[1]*0.5+0.5)
    x_center, y_center = det.Center

    x_ratio = (ref_w-x_center)/ref_w
    y_ratio = (ref_h-y_center)/ref_h

    w = int(det.Width/2+0.5)
    h = int(det.Height/2+0.5)

    x_shift = x_center+w*x_ratio
    y_shift = y_center+h*y_ratio
    w_shift = int(w*0.75+0.5)
    h_shift = int(h*0.75+0.5)


    if x_ratio >=0:
        p1 = [x_center-w,                  y_center-h]
        p2 = [x_center+int(w*x_ratio+0.5), y_center-h]
        p3 = [x_center+int(w*x_ratio+0.5), y_center+h]
        p4 = [x_center-w,                  y_center+h]
    else:
        #p1 = [x_center-int(w*x_ratio+0.5), y_center-h]
        p1 = [x_center                   , y_center-h]
        p2 = [x_center+w,                  y_center-h]
        p3 = [x_center+w,                  y_center+h]
        p4 = [x_center                   , y_center+h]


    p5 = [x_shift-w_shift/2, y_shift-h_shift/2]
    p6 = [x_shift+w_shift/2, y_shift-h_shift/2]
    p7 = [x_shift+w_shift/2, y_shift+h_shift/2]
    p8 = [x_shift-w_shift/2, y_shift+h_shift/2]

    pts = np.array([p1,p5,p8,p4], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(image,[pts],True,((51,255,51)), thickness=3)   

    pts = np.array([p1,p2,p6,p5], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(image,[pts],True,((51,255,51)), thickness=3) 

    pts = np.array([p2,p3,p7,p6], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(image,[pts],True,((51,255,51)), thickness=3) 

    pts = np.array([p7,p3,p4,p8], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(image,[pts],True,((51,255,51)), thickness=3)

    return image



def draw_boxes(detections, image, img_np):
    #print("debug: ", colors)
    for det in detections:
        cid    = det.ClassID

        if cid in target_list:
            left   = int(det.Left+0.5)
            top    = int(det.Top+0.5)
            right  = int(det.Right+0.5)
            bottom = int(det.Bottom+0.5)

            if cid in human_list:
                cls_name = "Person"
                cls_color = colors[0]
                #image[top:bottom, left:right, :] = img_np[top:bottom, left:right, :]
                cv2.rectangle(image, (left, top), (right, bottom), cls_color, 2)
            elif cid in car_list:
                cls_name = "Car"
                cls_color = colors[1]
                #image[top:bottom, left:right, :] = img_np[top:bottom, left:right, :]
                image = bbox_3D(det, image)
            elif cid in traffic_sign_list:
                cls_name = "Traffic Sign"
                cls_color = colors[2]

                image[top:bottom, left:right, :] = img_np[top:bottom, left:right, :]
                cv2.rectangle(image, (left, top), (right, bottom), cls_color, 2)

            cv2.putText(image, "[{}]".format(cls_name), (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cls_color, 2)
    return image



# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--det_network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--seg_network", type=str, default="fcn-resnet18-cityscapes-2048x1024", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 
parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
parser.add_argument("--visualize", type=str, default="overlay,mask", help="Visualization options (can be 'overlay' 'mask' 'overlay,mask'")
parser.add_argument("--ignore-class", type=str, default="void", help="optional name of class to ignore in the visualization results (default: 'void')")
parser.add_argument("--alpha", type=float, default=150.0, help="alpha blending value to use during overlay, between 0.0 and 255.0 (default: 150.0)")
parser.add_argument("--stats", action="store_true", help="compute statistics about segmentation mask class output")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
    opt = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# load the object detection network
det_net = jetson.inference.detectNet(opt.det_network, ['--network=ssd-inception-v2'], opt.threshold)
seg_net = jetson.inference.segNet(opt.seg_network, ['--network=fcn-resnet18-cityscapes-2048x1024'])

# set the alpha blending value
seg_net.SetOverlayAlpha(opt.alpha)

# create buffer manager
buffers = segmentationBuffers(seg_net, opt)

# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)


img_w, img_h = 1280, 720

img_res = np.zeros((img_h,img_w,3),dtype=np.uint8)
out = cv2.VideoWriter('lane_detection.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps=15, frameSize=(img_w,img_h*2))

# process frames until the user exits


detections=[]
num = 0


mot = MOTwithOpencv()
lane_det = RegionLaneDetectionwithOpencv()

while True:
    # capture the next image
    img = input.Capture()
    img_res.fill(0)
    
    img_np = jetson.utils.cudaAllocMapped(width=img.width, height=img.height, format=img.format)
    jetson.utils.cudaResize(img, img_np)
    img_np = jetson.utils.cudaToNumpy(img_np)
    jetson.utils.cudaDeviceSynchronize()


    black_lines_1, black_lines_2 = lane_det.detect(img_np)


    if num%10==0:
        detections = det_net.Detect(img, overlay=opt.overlay)
        jetson.utils.cudaDeviceSynchronize()

        if detections is not None:
            #multi_tracker = cv2.MultiTracker_create()
            mot.reset()
    
            for det in detections:
                left   = int(det.Left+0.5)
                top    = int(det.Top+0.5)
                right  = int(det.Right+0.5)
                bottom = int(det.Bottom+0.5)

                #print(img_np.shape)
                #print("left: {}   top: {}   right: {}    bottom: {}".format(left,top,right,bottom))
                #multi_tracker.add(trackerGen(6), img_np, (left,top,right-left,bottom-top))
                mot.addTracker(4, img_np, (left,top,right-left,bottom-top))

    else:
        #success, bboxes = multi_tracker.update(img_np);
        success, bboxes = mot.update(img_np);

        for i in range(0,len(detections)):
            detections[i].Left   = bboxes[i][0]
            detections[i].Top    = bboxes[i][1]
            detections[i].Right  = bboxes[i][0]+bboxes[i][2]
            detections[i].Bottom = bboxes[i][1]+bboxes[i][3]


    
    img_res = draw_boxes(detections,img_res, img_np)

    lanes = cv2.addWeighted(img_res, 0.8, black_lines_1, 1, 1)
    lanes = cv2.addWeighted(lanes, 0.8, black_lines_2, 1, 1)

    both = cv2.hconcat([img_np, lanes])
    out.write(both)

    # render the image
    both_cuda = jetson.utils.cudaFromNumpy(both)
    jetson.utils.cudaDeviceSynchronize()
    output.Render(both_cuda)
    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.det_network, det_net.GetNetworkFPS()))

    
    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break

    num+=1
