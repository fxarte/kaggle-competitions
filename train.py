from __future__ import print_function

import os
import sys
import re
import cv2
import numpy as np;
import csv
import mean_mask as my
import matplotlib.pylab as plt

data_dir='G:/data/ultrasound-nerve-segmentation'
bin_dir="E:/Tools/opencv/build/x64/vc14/bin"
csv_data=None
img_width=580
img_height=420
def load_csv_file():
  global csv_data
  if not csv_data:
    rows = []
    with open(data_dir+"/train_masks.csv", 'rb') as csvfile:
      csvreader = csv.reader(csvfile, delimiter=',')
      csvreader.next()
      for item in csvreader:
        # print(item)
        rows.append(item)
    csv_data = np.array(rows)

def get_pixels(subject_id,img_id):
  global csv_data
  for row in csv_data:
    if row[0] == subject_id and row[1] == img_id:
      return row[2]
      

def pixel_to_box(pixels):
  ''' Pixels run from top to bottom then to the right
  img_width=580
  img_height=420
  '''
  global img_width
  global img_height
  X1=0
  Y1=img_height
  X2=0
  Y2=0
  numbers=pixels.split()
  _x=0
  for i in range(0,len(numbers),2):
    pixel = int(numbers[i])
    _x=pixel/img_height
    
    #X1: set the x with the first pixel value
    if not X1:
      X1=_x
    # Y1
    _y=pixel%img_height
    if (_y < Y1):
      Y1 = _y
    
    run_length=int(numbers[i+1])
    #Y2 for Height
    if (_y + run_length > Y2):
      Y2=_y + run_length
  # X2 the last pixel coordinate
  X2=_x
  return X1,Y1, (X2-X1), (Y2-Y1)
    
def get_mask_box(subject_id,filename):
  m=re.search('{}_([0-9]+)'.format(subject_id), filename)
  img_id = m.group(1)
  pixels = get_pixels(str(subject_id), img_id)
  if pixels :
    return pixel_to_box(pixels)
  else:
    return None,None,None,None
def trace_box_imgs(img,img_mask,box, filename=""):
  edges = my.image_edges(img_mask)
  combined = my.image_with_mask(img, edges)
  x1,y1,w,h=box
  # print(x1, y1, w,h)
  x2 = x1+w
  y2 = y1+h
  cv2.rectangle(combined, (x1, y1), (x2, y2), (0,0,255), 1)
  cv2.imwrite(filename+".combined.png",combined)
  # cv2.imshow("combined", combined)
  # k = cv2.waitKey(0)

def generate_training_items():
  global data_dir
  global bin_dir
  global img_width
  global img_height
  train_folder='/train/'
  # train_folder='/tests/'
  dat_file_data=[]
  negatives=[]
  for subject_id in range(1, 2):
    for file in [f for f in os.listdir(data_dir+train_folder) if re.match(r'{}_[0-9]+\.tif'.format(subject_id), f)]:
      filename, file_extension = os.path.splitext(file)
      # print("Processing image "+file)
      raw_img_mask = data_dir+train_folder+filename+"_mask"+file_extension
      pos_img_path=data_dir+train_folder+file
      img=cv2.imread(pos_img_path,0)
      img_mask=cv2.imread(raw_img_mask,0)
      # height, width = cv2.imread(raw_img_mask,0).shape[:2]
      # print(width, height);
      x1,y1,w,h = get_mask_box(subject_id,filename)
      if x1:
        # print(x1,y1,w,h)
        # trace_box_imgs(img,img_mask,[x1,y1,w,h], filename=pos_img_path)
        dat_file_data.append("{} 1 {} {} {} {}".format("../train/"+file,x1,y1,w,h))
      else:
        negatives.append("{}".format("../train/"+file))

  #Creates the positives.dat file
  with open(data_dir+"/opencv/positives.dat", 'w') as f:
    for item in dat_file_data:
      f.write("%s\n" % item)
  #Creates the negatives.txt file
  with open(data_dir+"/opencv/negatives.txt", 'w') as f:
    for item in negatives:
      f.write("%s\n" % item)
  opencv_dir=data_dir+"/opencv"
  numPos=len(dat_file_data)
  numNeg=len(negatives)
  #Samples command
  print("{}/opencv_createsamples.exe -info {}/positives.dat -num {} -vec {}/positives.vec -w {} -h {}"
  .format(bin_dir, opencv_dir, numPos, opencv_dir, img_width, img_height))
  print()
  #Train command
  # opencv_traincascade -data "data" -vec "samples.vec" -bg "out_negatives.dat" -numPos 26000 -numNeg 4100 -numStages 16 -featureType LBP -w 20 -h 20 -bt GAB -minHitRate 0.995 -maxFalseAlarmRate 0.3 -weightTrimRate 0.95 -maxDepth 1 -maxWeakCount 100 -maxCatCount 256 -featSize 1
  print("{}/opencv_traincascade.exe -data {}/data -vec {}/positives.vec -bg {}/negatives.txt -numPos {} -numNeg {} -numStages 2 -w {} -h {}".format(bin_dir, opencv_dir, opencv_dir, opencv_dir, numPos,numNeg, img_width, img_height))
    
if __name__ == '__main__':
  from sys import argv
  load_csv_file()
  generate_training_items()
  # main(argv)
  #1 96
  # img=cv2.imread(data_dir+"/train/1_96.tif",0)
  # mask_img=cv2.imread(data_dir+"/train/1_96_mask.tif",0)
  # edges = my.image_edges(mask_img)
  # combined = my.image_with_mask(img, edges)
  # 328,66; 90x113
  # x1,y1,w,h=pixel_to_box("137896 17 138315 20 138731 27 139145 34 139557 45 139976 47 140392 52 140811 54 141228 58 141643 63 142061 67 142478 70 142897 72 143316 74 143733 77 144152 78 144571 79 144990 82 145409 83 145828 84 146247 86 146666 88 147084 90 147502 93 147920 95 148340 95 148758 98 149177 99 149596 100 150015 101 150434 103 150854 103 151273 104 151692 106 152111 107 152530 108 152950 108 153369 109 153789 109 154208 110 154628 110 155048 110 155468 110 155888 110 156308 110 156728 109 157148 109 157568 109 157988 109 158408 108 158828 108 159248 108 159669 106 160090 105 160511 103 160931 103 161352 101 161773 99 162193 99 162613 99 163034 97 163455 96 163876 94 164297 93 164718 91 165138 91 165559 89 165980 88 166401 86 166821 85 167242 84 167663 82 168084 79 168505 77 168926 74 169347 71 169768 68 170189 64 170610 61 171031 58 171452 56 171874 52 172295 48 172716 43 173137 41 173558 38 173980 34 174402 28 174824 23 175247 18")
  # print("box: {},{}->{},{} ".format(x1,y1,w,h))
  # x2 = x1+w
  # y2 = y1+h
  # cv2.rectangle(combined, (x1, y1), (x2, y2), (0,0,255), 1)
  # cv2.imwrite("my.png",img)
  # cv2.imshow("lalala", img)
  # cv2.imwrite("combined.png",combined)
  # cv2.imshow("combined", combined)
  
  # k = cv2.waitKey(0) # 0==wait forever