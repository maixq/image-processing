import math
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from PIL import Image, ImageDraw, ImageFilter
import cv2
import numpy as np
from shapely.geometry import Polygon
from refine_polygon import *
    
def get_pts(img_path: str):
    
    # read the input mask image
    img = cv2.imread(img_path)
    # img = cv2.bitwise_not(img)
    # convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply thresholding on the gray image to create a binary image
    ret,thresh = cv2.threshold(gray,127,255,0)

    # find the contours
    contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # take the first contour
    cnt = max(contours, key = cv2.contourArea)

    # compute the bounding rectangle of the contour
    x,y,w,h = cv2.boundingRect(cnt)

    # draw contour
    img = cv2.drawContours(img,[cnt],0,(0,255,255),2)

    # draw the bounding rectangle
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    
    # bbox 
    bbox = [[x,y+h],[x,y],[x+w, y], [x+w,y+h]]
    
    # compute rotated rectangle (minimum area)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    min_bbox = [x.tolist() for l in [box] for x in l]
    img = cv2.drawContours(img,[box],0,(0,255,255),2)
    
    # The extreme points
    l_m = list(cnt[cnt[:, :, 0].argmin()][0])
    r_m = list(cnt[cnt[:, :, 0].argmax()][0])
    t_m = list(cnt[cnt[:, :, 1].argmin()][0])
    b_m = list(cnt[cnt[:, :, 1].argmax()][0])
    pst = [l_m, r_m, t_m, b_m]

    # display(Image.fromarray(img))
    
    return img, bbox, min_bbox, pst

def get_slope(pt1: list, pt2: list):
    x1, y1 = pt1[0], pt1[1]
    x2, y2 = pt2[0], pt2[1]
    slope = (y1-y2)/(x1-x2)
    c = pt1[1]-slope*pt1[0]
    return slope, c

def get_polygon(min_bbox: list, bbox: list, extreme_pts: list, angle: str):
    polygon = []
    # unpack points
    l_m, r_m, t_m, b_m = extreme_pts
    bottomleft, topleft, topright, bottomright = bbox[0], bbox[1], bbox[2], bbox[3]
    left_vertical, right_vertical, bottom_horizontal = bottomleft[0], bottomright[0], bottomleft[1]
    # check if min_rotated_rec overlaps with bbox 
    if any(ele in bbox for ele in min_bbox):
        polygon = [l_m, bottomleft, b_m, [b_m[0]+150,b_m[1]],[r_m[0],(r_m[1]+b_m[1])/2], r_m]

    else:
        if angle == 'front-left':
            mtopleft, mtopright, mbottomright, mbottomleft = min_bbox[0], min_bbox[1], min_bbox[2], min_bbox[3]

            left_slope, left_slope_c = get_slope(mbottomleft, mtopleft)
            left_xinterset = (bottom_horizontal-left_slope_c) /left_slope
            polygon = [l_m, [int(left_xinterset),bottom_horizontal], b_m, r_m]
            
        if angle == 'front-right':
            mbottomleft, mtopleft, mtopright, mbottomright  = min_bbox[0], min_bbox[1], min_bbox[2], min_bbox[3]
            
            bottom_slope, bottom_slope_c = get_slope(mbottomright, mbottomleft)
            right_slope, right_slope_c = get_slope(mbottomright, mtopright)

            left_yintersect = bottom_slope*left_vertical+bottom_slope_c
            right_xintersect = (bottom_horizontal-right_slope_c) /right_slope
            polygon = [l_m,[left_vertical, left_yintersect], b_m,[b_m[0]+100,b_m[1]], [right_xintersect, bottom_horizontal], r_m]
        
        if angle == 'rear-right':
            mbottomleft, mtopleft, mtopright, mbottomright = min_bbox[0], min_bbox[1], min_bbox[2], min_bbox[3]
            polygon = [l_m, bbox[0], b_m,[b_m[0]+100,b_m[1]],[r_m[0], (b_m[1]+r_m[1])/2], r_m]
                    
    return polygon

def composite_imgs(foreground_path: str, xs: list, ys: list):
    foreground = Image.open(foreground_path).convert('RGBA')
    mask_shadow = Image.new('RGBA', foreground.size)
    pdraw = ImageDraw.Draw(mask_shadow)
    pdraw.polygon([(xs,ys) for xs,ys in zip(xs,ys)],
                  fill=(0,0,0,150),outline=(0,0,0,25))
    
    im_mask = mask_shadow.filter(ImageFilter.GaussianBlur(25))
    im_mask.paste(foreground, (0,10),foreground.convert('RGBA'))
    im_mask.save('res.png')
    return

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle")
    parser.add_argument("--mask")
    parser.add_argument("--foreground")
    args = parser.parse_args()
    
    angle = args.angle
    img_path = args.mask
    foreground_path = args.foreground
    
    img, bbox, min_bbox, pst = get_pts(img_path)
    
    polygon = get_polygon(min_bbox, bbox, pst, angle)
    obj = Object(polygon)
    smoothed_obj = obj.Smooth_by_Chaikin(number_of_refinements = 2, obj=obj)
    xs = [i for i,j in smoothed_obj]
    ys = [j for i,j in smoothed_obj]

    img = composite_imgs(foreground_path, xs, ys)