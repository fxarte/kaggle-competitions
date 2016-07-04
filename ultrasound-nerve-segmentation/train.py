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
csv_data=np.array([])
img_width=580
img_height=420
def load_csv_file():
  global csv_data
  if csv_data.size == 0:
    rows = []
    with open(data_dir+"/train_masks.csv", 'rb') as csvfile:
      csvreader = csv.reader(csvfile, delimiter=',')
      csvreader.next()
      for item in csvreader:
        # print(item)
        rows.append(item)
    csv_data = np.array(rows)

def get_pixels(subject_id,img_id):
  load_csv_file()
  global csv_data
  for row in csv_data:
    if row[0] == subject_id and row[1] == img_id:
      return row[2]
  return ""
      
def get_mean_box():
  box_means=[]
  train_folder='/train/'
  for subject_id in range(1, 48):
    for file in [f for f in os.listdir(data_dir+train_folder) if re.match(r'{}_[0-9]+\.tif'.format(subject_id), f)]:
      filename, file_extension = os.path.splitext(file)
      x1,y1,w,h = get_mask_box(subject_id,filename)
      if x1:
        box_means.append([x1,y1,w,h])
  
  # data=np.array(box_means)
  # np.save("box_data.npy", data)
  data=np.load("box_data.npy")
  plt.figure(figsize=(12, 12))
  plt.subplot(231)
  plt.scatter(data[:,2], data[:,3])
  plt.title("Widths vs Heights")

  plt.subplot(232)
  plt.scatter(data[:,0], data[:,1])
  plt.title("X vs Y")

  plt.subplot(233)
  plt.scatter(data[:,0], data[:,2])
  plt.title("X vs Widths")
  
  plt.subplot(234)
  plt.scatter(data[:,1], data[:,3])
  plt.title("Y vs Heights")
  
  plt.subplot(235)
  plt.scatter(data[:,0], data[:,3])
  plt.title("X vs Heigths")
  
  plt.subplot(236)
  plt.scatter(data[:,1], data[:,2])
  plt.title("Y vs Widths")
  
  plt.show()
  print(np.mean(np.dstack(box_means), axis=2))
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
  pad=20
  return X1-pad,Y1-pad, (X2-X1)+2*pad, (Y2-Y1)+2*pad
    
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
def get_adjacent_box(box,direction):
  x,y,w,h=box
  X1=x
  Y1=y
  W1=w
  H1=h
  if direction=="N":
    X1=x-w
    Y1=y-h
    W1=3*w
  elif direction=="S":
    X1=x-w
    Y1=y+h
    W1=3*w
  elif direction=="E":
    X1=x+w
    Y1=y-h
    H1=3*h
  elif direction=="W":
    X1=x-w
    Y1=y-h
    H1=3*h
  return X1,Y1,W1,H1

def save_dat_file(dat_file_data,filename):
  with open(data_dir+"/opencv/"+filename, 'w') as f:
    for item in dat_file_data:
      f.write("%s\n" % item)
  
def generate_training_items():
  global data_dir
  global bin_dir
  global img_width
  global img_height
  train_folder='/train/'
  # train_folder='/tests/'
  dat_file_data=[]
  dat_file_data_N=[]
  dat_file_data_S=[]
  dat_file_data_E=[]
  dat_file_data_W=[]
  negatives=[]
  good_possitives=[1,5,10,14,21,25,26,29,32,41,43,47]
  box_means=[]
  ADJACENT=False
  for subject_id in range(1, 48):
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
      # if subject_id>5:
        # break
      if x1:
        # print(x1,y1,w,h)
        trace_box_imgs(img,img_mask,[x1,y1,w,h], filename=data_dir+"/output/"+file+".POS.png")
        if True or subject_id in good_possitives:
          box_means.append([x1,y1,w,h])
          dat_file_data.append("{} 1 {} {} {} {}".format("../train/"+file,x1,y1,w,h))
          if ADJACENT:
            _X,_Y,_W,_H=get_adjacent_box([x1,y1,w,h], "N")
            dat_file_data_N.append("{} 1 {} {} {} {}".format("../train/"+file,_X,_Y,_W,_H))
            _X,_Y,_W,_H=get_adjacent_box([x1,y1,w,h], "S")
            dat_file_data_S.append("{} 1 {} {} {} {}".format("../train/"+file,_X,_Y,_W,_H))
            _X,_Y,_W,_H=get_adjacent_box([x1,y1,w,h], "E")
            dat_file_data_E.append("{} 1 {} {} {} {}".format("../train/"+file,_X,_Y,_W,_H))
            _X,_Y,_W,_H=get_adjacent_box([x1,y1,w,h], "W")
            dat_file_data_W.append("{} 1 {} {} {} {}".format("../train/"+file,_X,_Y,_W,_H))

      else:
        trace_box_imgs(img,img_mask,[0,0,1,1], filename=data_dir+"/output/"+file+".NEG.png")
        negatives.append("{}".format("../train/"+file))

  # return

  #Creates the positives.dat file
  with open(data_dir+"/opencv/positives.dat", 'w') as f:
    for item in dat_file_data:
      f.write("%s\n" % item)
  if ADJACENT:
    save_dat_file(dat_file_data_N, "positives_n.dat")
    save_dat_file(dat_file_data_S, "positives_s.dat")
    save_dat_file(dat_file_data_E, "positives_e.dat")
    save_dat_file(dat_file_data_W, "positives_w.dat")
  
  #Creates the negatives.txt file
  with open(data_dir+"/opencv/negatives.txt", 'w') as f:
    for item in negatives:
      f.write("%s\n" % item)
  opencv_dir=data_dir+"/opencv"
  numPos=len(dat_file_data)
  numNeg=len(negatives)
  window_w=130
  window_h=160
  #Samples command
  roations="-maxxangle 0.1 -maxyangle 0 -maxzangle 0.1"
  bgcolor="-bgcolor 0 -bgthresh 8"
  print("{}/opencv_createsamples.exe -info {}/positives.dat -num {} -vec {}/positives.vec {} {} -w {} -h {}"
  .format(bin_dir, opencv_dir, numPos, opencv_dir, bgcolor, roations, window_w, window_h))
  
  if ADJACENT:
    print("------ Adjacebnt regions -----")
    print("{}/opencv_createsamples.exe -info {}/positives_n.dat -num {} -vec {}/positives_n.vec -w {} -h {}".format(bin_dir, opencv_dir, numPos, opencv_dir, window_w, window_h))
    print("{}/opencv_createsamples.exe -info {}/positives_s.dat -num {} -vec {}/positives_s.vec -w {} -h {}".format(bin_dir, opencv_dir, numPos, opencv_dir, window_w, window_h))
    print("{}/opencv_createsamples.exe -info {}/positives_e.dat -num {} -vec {}/positives_e.vec -w {} -h {}".format(bin_dir, opencv_dir, numPos, opencv_dir, window_w, window_h))
    print("{}/opencv_createsamples.exe -info {}/positives_w.dat -num {} -vec {}/positives_w.vec -w {} -h {}".format(bin_dir, opencv_dir, numPos, opencv_dir, window_w, window_h))

  #Train command
  stages=5
  minHitRate=.99
  _trainNumPos=int(numPos*minHitRate)
  print("cd {}\n")
  # opencv_traincascade -data "data" -vec "samples.vec" -bg "out_negatives.dat" -numPos 26000 -numNeg 4100 -numStages 16 -featureType LBP -w 20 -h 20 -bt GAB -minHitRate 0.995 -maxFalseAlarmRate 0.3 -weightTrimRate 0.95 -maxDepth 1 -maxWeakCount 100 -maxCatCount 256 -featSize 1
  print("{}/opencv_traincascade.exe -data data -vec positives.vec -bg negatives.txt -numPos {} -numNeg {} -numStages {} -minHitRate {} -featureType LBP -w {} -h {}"
  .format(bin_dir, _trainNumPos,numNeg,stages,minHitRate, window_w, window_h))
  
  if ADJACENT:
    print("------ Adjacebnt regions -----")
    print("{}/opencv_traincascade.exe -data data -vec positives_n.vec -bg negatives.txt -numPos {} -numNeg {} -numStages {} -minHitRate {} -featureType LBP -w {} -h {}".format(bin_dir, _trainNumPos,numNeg,stages,minHitRate, window_w, window_h))
    print("{}/opencv_traincascade.exe -data data -vec positives_s.vec -bg negatives.txt -numPos {} -numNeg {} -numStages {} -minHitRate {} -featureType LBP -w {} -h {}".format(bin_dir, _trainNumPos,numNeg,stages,minHitRate, window_w, window_h))
    print("{}/opencv_traincascade.exe -data data -vec positives_e.vec -bg negatives.txt -numPos {} -numNeg {} -numStages {} -minHitRate {} -featureType LBP -w {} -h {}".format(bin_dir, _trainNumPos,numNeg,stages,minHitRate, window_w, window_h))
    print("{}/opencv_traincascade.exe -data data -vec positives_w.vec -bg negatives.txt -numPos {} -numNeg {} -numStages {} -minHitRate {} -featureType LBP -w {} -h {}".format(bin_dir, _trainNumPos,numNeg,stages,minHitRate, window_w, window_h))

if __name__ == '__main__':
  from sys import argv
  load_csv_file()
  # get_mean_box()
  generate_training_items()
  # main(argv)
