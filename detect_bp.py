from __future__ import division
import numpy as np
import cv2
import os
import re
import cv2
import mean_mask as my
import train
import sys
import pprint

IMGS_BASE_DIR='G:/data/ultrasound-nerve-segmentation'
DATA_DIR='G:/data/ultrasound-nerve-segmentation'
ultrasound_bp_cascade_nerve_path=IMGS_BASE_DIR+'/opencv/models_cnsew_90x100/data/cascade.xml'
ultrasound_bp_cascade_nerve_N_path=IMGS_BASE_DIR+'/opencv/models_cnsew_90x100/data_n/cascade.xml'
ultrasound_bp_cascade_nerve_S_path=IMGS_BASE_DIR+'/opencv/models_cnsew_90x100/data_s/cascade.xml'
ultrasound_bp_cascade_nerve_E_path=IMGS_BASE_DIR+'/opencv/models_cnsew_90x100/data_e/cascade.xml'
ultrasound_bp_cascade_nerve_W_path=IMGS_BASE_DIR+'/opencv/models_cnsew_90x100/data_w/cascade.xml'

ultrasound_bp_cascade_nerve_path=IMGS_BASE_DIR+'/opencv/models_c_n_s_e_w_75x75/cascade_nerve_nerve.xml'
ultrasound_bp_cascade_nerve_N_path=IMGS_BASE_DIR+'/opencv/models_c_n_s_e_w_75x75/cascade_nerve_n.xml'
ultrasound_bp_cascade_nerve_S_path=IMGS_BASE_DIR+'/opencv/models_c_n_s_e_w_75x75/cascade_nerve_s.xml'
ultrasound_bp_cascade_nerve_E_path=IMGS_BASE_DIR+'/opencv/models_c_n_s_e_w_75x75/cascade_nerve_e.xml'
ultrasound_bp_cascade_nerve_W_path=IMGS_BASE_DIR+'/opencv/models_c_n_s_e_w_75x75/cascade_nerve_w.xml'



ultrasound_bp_test=IMGS_BASE_DIR+'/tests/1_2.tif'
ultrasound_bp_cascade_nerve = cv2.CascadeClassifier(ultrasound_bp_cascade_nerve_path)
ultrasound_bp_cascade_nerve_n = cv2.CascadeClassifier(ultrasound_bp_cascade_nerve_N_path)
ultrasound_bp_cascade_nerve_s = cv2.CascadeClassifier(ultrasound_bp_cascade_nerve_S_path)
ultrasound_bp_cascade_nerve_e = cv2.CascadeClassifier(ultrasound_bp_cascade_nerve_E_path)
ultrasound_bp_cascade_nerve_w = cv2.CascadeClassifier(ultrasound_bp_cascade_nerve_W_path)
pp = pprint.PrettyPrinter(indent=4)
slack=0
DEBUG=True
DEBUG=False
colors = {"c":(255,0,0), "n":(0,255,0), "s":(255,255,255), "e":(0,255,255), "w":(0,0,255)}
regions = {"c":"center", "n":"north", "s":"south", "e":"east", "w":"west"}

def merge_imgs(imgs, img_id,subject_id):
  global pp
  global colors
  global regions

  gray=imgs["gray"]
  mask=imgs["mask"]
  edges = my.image_edges(mask)
  combined = my.image_with_mask(gray, edges)
  c = 0
  font = cv2.FONT_HERSHEY_PLAIN
  
  for region,boxes in imgs["boxes"].iteritems():
    count=boxes.shape[0]
    color = colors[region]
    # print("region: {}, y: {}\n".format(regions[region], color))
    for pos in range(0,boxes.shape[0]):
      # sys.stdout.write('.')
      box=boxes[pos]
      x,y,w,h = box
      # color = (255,255,0)
      r=2*c+(w/2)
      cy=y+h/2
      cx=x+r
      # cv2.circle(combined,(cx,cy),r,color,1)
      # cv2.PutText(combined, regions[c], (5,10+(c*10)), cv2.CV_FONT_HERSHEY_SIMPLEX,1, color) 
      x2 = int(x+w+c*2)
      y2 = int(y+h+c*2)
      cv2.rectangle(combined,(int(x),int(y)),(x2,y2),color,1)
    cv2.putText(combined,regions[region]+": "+str(count),(10,20+(c*20)), font, 1,color,2)
    c +=1
  cv2.imwrite("test/"+subject_id+"__"+img_id+"__POS.combined.png",combined);
    
    
def get_valid_region_boxes(box, boxes_region, region):
  # TODO Move this checks to a proper loop
  global slack
  global pp
  global DEBUG
  # print("BOxes for {} : {}".format(region, boxes_region))
  x,y,w,h = box
  # print(x,y,w,h)
  valid_boxes=[]
  if DEBUG or not boxes_region.all():
    return boxes_region
  else:
    if boxes_region.shape[0]>0:
      if region=="n":
        for x1,y1,w1,h1 in boxes_region:
          if y1<(y-(h+h*slack)):
            valid_boxes.append((x1,y1,w1,h1))
      elif region=="s":
        for x1,y1,w1,h1 in boxes_region:
          if y1>(y+(h+h*slack)):
            valid_boxes.append((x1,y1,w1,h1))
      elif region=="e":
        for x1,y1,w1,h1 in boxes_region:
          if x1>(x+(w+w*slack)):
            valid_boxes.append((x1,y1,w1,h1))
      elif region=="w":
        for x1,y1,w1,h1 in boxes_region:
          if x1<(x-(w+w*slack)):
            valid_boxes.append((x1,y1,w1,h1))
    return valid_boxes
    
def is_box_in_list(list, box):
  for i in range(0,len(list)):
    if box == list[i]:
      return True
  return False


def is_BP_nerve(img_gray):
  global ultrasound_bp_cascade_nerve
  global ultrasound_bp_cascade_nerve_n
  global ultrasound_bp_cascade_nerve_s
  global ultrasound_bp_cascade_nerve_e
  global ultrasound_bp_cascade_nerve_w
  global pp
  boxes = dict()
  boxes["boxes"]={}
  boxes["boxes"]["n"]=[]
  boxes["boxes"]["s"]=[]
  boxes["boxes"]["e"]=[]
  boxes["boxes"]["w"]=[]
  
  # _minSize=(90,100)
  # _maxSize=(90,100)
  _minSize=(40,50)
  _maxSize=(90,100)
  # http://stackoverflow.com/a/20805153
  boxes_c = ultrasound_bp_cascade_nerve.detectMultiScale(img_gray,scaleFactor=1.05,minNeighbors=5,minSize=_minSize, maxSize=_maxSize)

  # _minSize=(190,100)
  # _maxSize=(250,100)
  boxes_n=ultrasound_bp_cascade_nerve_n.detectMultiScale(img_gray,scaleFactor=1.05,minNeighbors=5,minSize=_minSize, maxSize=_maxSize)

  # _minSize=(190,100)
  # _maxSize=(250,100)
  boxes_s=ultrasound_bp_cascade_nerve_s.detectMultiScale(img_gray,scaleFactor=1.05,minNeighbors=5,minSize=_minSize, maxSize=_maxSize)

  # _minSize=(90,250)
  # _maxSize=(90,300)
  boxes_e=ultrasound_bp_cascade_nerve_e.detectMultiScale(img_gray,scaleFactor=1.05,minNeighbors=5,minSize=_minSize, maxSize=_maxSize)

  # _minSize=(90,250)
  # _maxSize=(90,300)
  boxes_w=ultrasound_bp_cascade_nerve_w.detectMultiScale(img_gray,scaleFactor=1.05,minNeighbors=5,minSize=_minSize, maxSize=_maxSize)
  
  for b in range(0, len(boxes_c)):
    votes=0
    # print("BOX#:{}".format(b))
    valid_boxes=None
    valid_boxes = get_valid_region_boxes(boxes_c[b], boxes_n, "n")
    if(len(valid_boxes)>0):
      boxes["boxes"]["n"] = valid_boxes
      votes +=1
    valid_boxes=None
    valid_boxes = get_valid_region_boxes(boxes_c[b], boxes_s, "s")
    if(len(valid_boxes)>0):
      boxes["boxes"]["s"] = valid_boxes
      votes +=1
    valid_boxes=None
    valid_boxes = get_valid_region_boxes(boxes_c[b], boxes_e, "e")
    if(len(valid_boxes)>0):
      boxes["boxes"]["e"] = valid_boxes
      votes +=1
    valid_boxes=None
    valid_boxes = get_valid_region_boxes(boxes_c[b], boxes_w, "w")
    if(len(valid_boxes)>0):
      boxes["boxes"]["w"] = valid_boxes
      votes +=1
    valid_boxes=None
    if votes>3:
      boxes["boxes"]["c"]=np.array((boxes_c[b],))

  boxes["boxes"]["n"]=np.array(boxes["boxes"]["n"])
  boxes["boxes"]["s"]=np.array(boxes["boxes"]["s"])
  boxes["boxes"]["e"]=np.array(boxes["boxes"]["e"])
  boxes["boxes"]["w"]=np.array(boxes["boxes"]["w"])
  # pp.pprint(boxes)
  return boxes
  
    

def main(argv):
  train_folder='/train/'
  output='/output/'
  #Determine train error
  #subject_id
  positives=0
  true_positives=0
  false_positives=0
  negatives=0
  true_negatives=0
  false_negatives=0
  c = 0
  positives=0
  negatives=0
  for file in [f for f in os.listdir(DATA_DIR+train_folder)]:
    filename, file_extension = os.path.splitext(file)
    split_filename=filename.split('_')
    if len(split_filename)==3:
      continue
    # print("..................................................")
    img_id,subject_id=split_filename[0],split_filename[1]
    
    pixels = train.get_pixels(str(subject_id), img_id)
    positive_truth=0
    negative_truth=0
    if pixels:
      positive_truth=1
    else:
      negative_truth=1
    
    
    raw_img_mask = DATA_DIR+train_folder+filename+"_mask"+file_extension
    pos_img_path=DATA_DIR+train_folder+file
    output_img_path=DATA_DIR+output+file
    img_gray=cv2.imread(pos_img_path,0)
    
    is_nerve = is_BP_nerve(img_gray)
    img_mask=cv2.imread(raw_img_mask,0)
    mask_edges = my.image_edges(img_mask)
    # sys.stdout.write(str(c))
    if c >= 100:
      print("Stopping")
      break
    if "c" in is_nerve["boxes"]:
      is_nerve["mask"] = mask_edges
      is_nerve["gray"] = img_gray
      merge_imgs(is_nerve,img_id,subject_id )
      sys.stdout.write("+")
      c +=1
      positives+=1
      true_positives +=positive_truth
      if negative_truth:
        false_negatives +=1
      continue
    else:
      negatives+=1
      true_negatives +=negative_truth
      if positive_truth:
        false_positives +=1
      sys.stdout.write("-")
      combined = my.image_with_mask(img_gray, mask_edges)
      cv2.imwrite("test/"+subject_id+"__"+img_id+"__NEG.combined.png",combined);
      continue
    '''
    
    

    if len(nerve_bp_list)>0:
      # cluster similar rectangles
      print(len(nerve_bp_list))
      # nerve_bp_list,weights = cv2.groupRectangles(list(nerve_bp),2,1)
      # print(nerve_bp_list,weights)
      # print(len(nerve_bp_list))
      # if (len(nerve_bp_list)==0):
        # continue
      # print(nerve_bp_list,weights)
      # print("------------")
      # return
      # nerve_bp_list=nerve_bp
      positives +=1
      if (positive_truth):
        true_positives +=1
      x,y,w,h=nerve_bp_list[0]
      # cv2.rectangle(img_gray,(x,y),(x+w,y+h),(0,255,0),2)
      r=w/2
      cy=y+h/2
      cx=x+r
      # print(r,cx,cy)
      cv2.circle(img_gray,(cx,cy),r,(255,0,0),2)
      # Positive
      # x1,y1,w,h = pixel_to_box(pixels)
      # cv2.rectangle(img_gray,(x1,y1),(x+w,y+h),(255,0,0),2)
      #generate a merged image
      img_mask=cv2.imread(raw_img_mask,0)
      mask_edges = my.image_edges(img_mask)
      combined = my.image_with_mask(img_gray, mask_edges)
      cv2.imwrite(output_img_path+".combined.png",combined);
    else:
      negatives +=1
      if negative_truth:
        true_negatives+=1
    c +=1
    # if c==15:
      # return
    '''
  # true_positives="N/A"
  # true_negatives="N/A"
  print("Positives:{}, Negatives:{}, True positives: {}, False positives: {}, true negatives:{}, false negatives: {}".format(positives, negatives, true_positives, false_positives, true_negatives, false_negatives))
  print("Presicion: {}, recall:{}, F-1: ", (true_positives/(true_positives+false_positives)),(true_positives/(false_negatives+true_positives)))

def test():
  img = cv2.imread(ultrasound_bp_test)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  nerve = ultrasound_bp_cascade_nerve.detectMultiScale(
      gray,
      scaleFactor=1.1,
      minNeighbors=1,
      minSize=(70, 70),
      maxSize=(90, 90),
      flags = cv2.CASCADE_SCALE_IMAGE
  )
  # print nerve[0]#, nerve[1], nerve[2]
  for (x,y,w,h) in nerve:
      cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
      # roi_gray = gray[y:y+h, x:x+w]
      # roi_color = img[y:y+h, x:x+w]
      # eyes = eye_cascade.detectMultiScale(roi_gray)
      # for (ex,ey,ew,eh) in eyes:
          # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

      cv2.imshow('img',img)
      cv2.waitKey(0)
      cv2.destroyAllWindows()


if __name__ == '__main__':
  from sys import argv
  main(argv)