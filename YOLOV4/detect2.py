import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
from PIL import Image
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *



def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz, cfg, names = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = Darknet(cfg, imgsz).cuda()
    try:
        model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
        #model = attempt_load(weights, map_location=device)  # load FP32 model
        #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    except:
        load_darknet_weights(model, weights[0])
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, auto_size=64)

    # Get names and colors
    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    strings = [] #多數決存放的string
    bestimg=0
    acccount=0
    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
           
           
            

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    if save_img:# 裁剪檢測框區域
                        x1, y1, x2, y2 = map(int, xyxy)
                        cropped_img = im0[y1:y2, x1:x2]
                         # 儲存裁剪後的圖片
                        save_cropped_path = save_path.replace('.jpg', f'_{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}.jpg')
                        cv2.imwrite(save_cropped_path, cropped_img)
                        gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                        _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                        denoised_img = cv2.fastNlMeansDenoising(binary_img, None, 10, 7, 21)
                        enhanced_img = cv2.equalizeHist(denoised_img)# enhanced_img = cv2.bitwise_not(enhanced_img)
                        pil_img = Image.fromarray(enhanced_img)  
                        text = pytesseract.image_to_string(pil_img, lang='eng', config='--psm 6 --oem 1 ')  # 英文文字檢測
                        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                        text_filtered = ''.join(filter(allowed_chars.__contains__, text)).upper()
                        #print("Detected Text:", text_filtered)
                        cv2.putText(im0, text_filtered, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    def calculate_formula_result(letters):
                        values = {
                                    'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17, 'H': 18, 'I': 19, 'J': 20,
                                    'K': 21, 'L': 23, 'M': 24, 'N': 25, 'O': 26, 'P': 27, 'Q': 28, 'R': 29, 'S': 30, 'T': 31,
                                    'U': 32, 'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38, '1': 1,  '2':  2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9
                                }
                        total = 0
                        for i, letter in enumerate(letters):
                            if letter in values:
                                total += values[letter] * (2 ** i)
                        return total % 11  
                    


                    # path1 ="C:/Users/p9514/YOLOv4/data/output/SEKU5875349"
                    # countimg = 0
                    # for i in os.listdir(path1):
                    #     countimg = countimg+1  
                    
                    def find_most_common_char(string):   # 多數決
                        char_count = {}
                        for char in string:
                            if char in char_count:
                                char_count[char] += 1
                            else:
                                char_count[char] = 1

                        max_count = 0
                        most_common_char = ''

                        for char, count in char_count.items():
                            if count > max_count:
                                max_count = count
                                most_common_char = char
                                bestimg = max_count

                            # if bestimg >0 :
                            
                            #     print("績效為:",bestimg/countimg)
                            #     break
                        return most_common_char,max_count
                    


                    
                        
                        
                    input_letters = text_filtered
                    input_letters = input_letters[:10]
                    # result = calculate_formula_result(input_letters)
                    # print("Detected Text:", input_letters)
                    # print(result,type(result))
                    print("讀取圖片名稱:",save_path[-15:-4])
                    # if  result == int(save_path[-5]):
                    print("偵測貨櫃號:", input_letters)
                    # print(result)
                    
                    
                    if input_letters==save_path[-15:-5]:
                        acccount=acccount+1
                    print("預測績效:",acccount/35)
                    


                    
                            
                # print(strings)
                    # print("出現最多次數的貨號",find_most_common_char(strings))
                        
                        

                    


                
                
                    # if view_img:# 裁剪檢測框區域
                    #     x1, y1, x2, y2 = map(int, xyxy)
                    #     cropped_view = im0[y1:y2, x1:x2]
                    #      # 儲存裁剪後的圖片
                    #     save_cropped_path = save_path.replace('.mkv', f'_{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}.mkv')
                    #     cv2.imwrite(save_cropped_path, cropped_view)
                    #     gray_img = cv2.cvtColor(cropped_view, cv2.COLOR_BGR2GRAY)
                    #     _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    #     denoised_img = cv2.fastNlMeansDenoising(binary_img, None, 10, 7, 21)
                    #     enhanced_img = cv2.equalizeHist(denoised_img)# enhanced_img = cv2.bitwise_not(enhanced_img)
                    #     pil_img = Image.fromarray(enhanced_img)  
                    #     text = pytesseract.image_to_string(pil_img, lang='eng')  # 英文文字檢測
                    #     allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    #     text_filtered = ''.join(filter(allowed_chars.__contains__, text)).upper()
                    #     print("Detected Text:", text_filtered)
                    #     cv2.putText(im0, text_filtered, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                if save_img or view_img:  # Add bbox to image
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov4.weights', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='models/yolov4.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
