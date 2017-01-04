# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import print_function
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# import matplotlib.pylab as plt
import os
import sys
import re
import cv2

def image_with_mask(img, mask):
  # returns a copy of the image with edges of the mask added in red
  img_color = grays_to_RGB(img)
  # img_color=img
  mask_edges = cv2.Canny(mask, 100, 200) > 0
  img_color[mask_edges, 0] = 255  # set channel 0 to bright red, green & blue channels to 0
  img_color[mask_edges, 1] = 0
  img_color[mask_edges, 2] = 0
  return img_color


def rgb2gray(rgb):
  w=Image.fromarray(rgb,mode='L')
  return w

def image_edges(color_img):
  img = np.uint8(color_img)
  ret2,detected_edges = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  return cv2.Canny(detected_edges, 0.1,1)
  
def grays_to_RGB(img):
  # turn 2D grayscale image into grayscale RGB
  return np.dstack((img, img, img))
  
def mask_not_blank(mask):
  return sum(mask.flatten()) > 0

def numpy_image(data,title=None, save_to=None):
  img = Image.fromarray(data)
  if img.mode != 'RGB':
    img = img.convert('RGB')
  img.save(save_to)

def plot_image(img, title=None, save_to=None):
  plt.figure(figsize=(15,20))
  plt.title(title)
  plt.imshow(img)
  if save_to:
    plt.savefig(save_to, bbox_inches='tight')
  else:
    plt.show()
  plt.close()

def main():
  data_dir='G:/data/ultrasound-nerve-segmentation'
  for i in range(1, 48):
    pos=[]
    neg=[]
    pos_mask=[]
    i_msk=[]
    msks=[]
    all_edges=[]
    
    
    for file in [f for f in os.listdir(data_dir+'/train/') if re.match(r'{}_[0-9]+\.tif'.format(i), f)]:
      filename, file_extension = os.path.splitext(file)
      # print(filename+" "+filename+"_mask"+file_extension);
      # sys.exit()
      raw_img = plt.imread(data_dir+"/train/"+file)
      # im = np.array(raw_img)
      im = raw_img
      raw_img_mask = plt.imread(data_dir+"/train/"+filename+"_mask"+file_extension)
      # im_mask = np.array(raw_img_mask)
      im_mask = raw_img_mask
      edges = image_edges(raw_img)
      all_edges.append(edges)
      # numpy_image(edges,save_to=data_dir+"/output/{}.png".format(filename))
      
      if mask_not_blank(im_mask):
        pos.append(im)
        pos_mask.append(im_mask)
        combined = image_with_mask(raw_img, raw_img_mask)
        # numpy_image(combined,save_to=data_dir+"/output/{}_masked.png".format(filename)) 
        # numpy_image(raw_img,save_to=data_dir+"/output/{}_pos.png".format(filename)) 
        i_msk.append(combined)
        msks.append(image_edges(raw_img_mask))
        # plot_image(combined)
      else:
        neg.append(im)
        # numpy_image(raw_img,save_to=data_dir+"/output/{}_neg_raw.png".format(filename))
          
    # print("\n")
    # pos = np.sum(np.dstack(pos), axis=2)
    pos = np.mean(np.dstack(pos), axis=2)
    neg = np.mean(np.dstack(neg), axis=2)
    pos_mask = np.mean(np.dstack(pos_mask), axis=2)
    i_msk = np.mean(np.dstack(i_msk), axis=2)
    msks_mean = np.mean(np.dstack(msks), axis=2)
    edges_mean = np.mean(np.dstack(all_edges), axis=2)
    
    # numpy_image(msks_mean,save_to=data_dir+"/means/{}_masks.png".format(i))
    # numpy_image(edges_mean,save_to=data_dir+"/means/{}_edges.png".format(i))
    # numpy_image(pos,save_to=data_dir+"/means/{}pos_raw.png".format(i))
    numpy_image(pos_mask,save_to=data_dir+"/means/{}pos_mask_raw.png".format(i))
    # numpy_image(neg,save_to=data_dir+"/means/{}neg_raw.png".format(i))

    p=[]
    p = np.sum(np.dstack([pos,pos_mask]), axis=2)
    # plot_image(i_msk, title="mask patient: {}".format(i), save_to=data_dir+"/output/patient_{}.png".format(i))

    # m=image_with_mask(pos, pos_mask)
    # numpy_image(image_edges(pos), title="patient: {}".format(i), save_to=data_dir+"/output/patient_{}.png".format(i))
    # numpy_image(image_edges(pos_mask), title="patient mask: {}".format(i), save_to=data_dir+"/output/patient_{}_mask.png".format(i))
    
    
if __name__ == '__main__':
  from sys import argv
  main(argv)