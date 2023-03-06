import math
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from PIL import Image, ImageDraw, ImageFilter
import cv2
import numpy as np
from shapely.geometry import Polygon

def Sum_points(P1, P2):
    x1, y1 = P1
    x2, y2 = P2
    return x1+x2, y1+y2

def Multiply_point(multiplier, P):
    x, y = P
    return float(x)*float(multiplier), float(y)*float(multiplier)

def Check_if_object_is_polygon(Cartesian_coords_list):
    if Cartesian_coords_list[0] == Cartesian_coords_list[len(Cartesian_coords_list)-1]:
        return True
    else:
        return False

class Object():

    def __init__(self, Cartesian_coords_list):
        self.Cartesian_coords_list = Cartesian_coords_list

    def Find_Q_point_position(self, P1, P2):
        Summand1 = Multiply_point(float(3)/float(4), P1)
        Summand2 = Multiply_point(float(1)/float(4), P2)
        Q = Sum_points(Summand1, Summand2) 
        return Q

    def Find_R_point_position(self, P1, P2):
        Summand1 = Multiply_point(float(1)/float(4), P1)
        Summand2 = Multiply_point(float(3)/float(4), P2)        
        R = Sum_points(Summand1, Summand2)
        return R

    def Smooth_by_Chaikin(self, number_of_refinements):
        refinement = 1
        copy_first_coord = Check_if_object_is_polygon(self.Cartesian_coords_list)
        while refinement <= number_of_refinements:
            self.New_cartesian_coords_list = []

            for num, tuple in enumerate(self.Cartesian_coords_list):
                if num+1 == len(self.Cartesian_coords_list):
                    pass
                else:
                    P1, P2 = (tuple, self.Cartesian_coords_list[num+1])
                    Q = obj.Find_Q_point_position(P1, P2)
                    R = obj.Find_R_point_position(P1, P2)
                    self.New_cartesian_coords_list.append(Q)
                    self.New_cartesian_coords_list.append(R)

            if copy_first_coord:
                self.New_cartesian_coords_list.append(self.New_cartesian_coords_list[0])

            self.Cartesian_coords_list = self.New_cartesian_coords_list
            refinement += 1
        return self.Cartesian_coords_list

    
def get_pts(img_path: str, dir_path: str):
    
    # read the input mask image
    img = cv2.imread(dir_path+img_path)
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
    xcor = [p[0] for p in pst]
    ycor = [p[1] for p in pst]
    
    # display(Image.fromarray(img))
    
    return img, bbox, min_bbox, pst

def get_linear(mbottomleft, mtopleft, mtopright, mbottomright, bottom_horizontal, right_vertical, left_vertical):
        bottom_slope =  (mbottomright[1]-mbottomleft[1])/(mbottomright[0]-mbottomleft[0])
        bottom_slope_c = mbottomright[1]-(bottom_slope*mbottomright[0])
        right_yintersect = bottom_slope*right_vertical+bottom_slope_c
        
        left_slope = (mtopleft[1]-mbottomleft[1])/(mtopleft[0]-mbottomleft[0])
        left_slope_c = mbottomleft[1]-(left_slope*mbottomleft[0])
        left_xinterset = (bottom_horizontal-left_slope_c) /left_slope
        
        right_slope = (mtopright[1]-mbottomright[1])/(mtopright[0]-mbottomright[0])
        right_slope_c = mbottomright[1]-(right_slope*mbottomright[0])
        right_xinterset = (bottom_horizontal-right_slope_c) /right_slope
        
        return bottom_slope, bottom_slope_c, left_slope, left_slope_c, right_slope, right_slope_c


def get_polygon(min_bbox, bbox, pst, angle, bottom_horizontal, right_vertical, left_vertical):
    # check if min_rotated_rec overlaps with bbox 
    if any(ele in bbox for ele in min_bbox):
        polygon_pts = [l_m, bottomleft, b_m, [b_m[0]+150,b_m[1]],[r_m[0],(r_m[1]+b_m[1])/2], r_m]

    else:
        if angle == 'front-left':
            mtopleft, mtopright, mbottomright, mbottomleft = min_bbox[0], min_bbox[1], min_bbox[2], min_bbox[3]
            bottom_slope, bottom_slope_c, left_slope, left_slope_c, right_slope, right_slope_c = get_linear(mbottomleft, mtopleft, mtopright, mbottomright, bottomright[1], bottomright[0], bottomleft[0])
            left_xinterset = (bottom_horizontal-left_slope_c) /left_slope
            polygon = [l_m, [int(left_xinterset),bottom_horizontal], b_m,  [right_vertical,int(right_yintersect)], r_m]
            
        if angle == 'front-right':
            mbottomleft, mtopleft, mtopright, mbottomright  = min_bbox[0], min_bbox[1], min_bbox[2], min_bbox[3]
            bottom_slope, bottom_slope_c, left_slope, left_slope_c, right_slope, right_slope_c = get_linear(mbottomleft, mtopleft, mtopright, mbottomright, bottomright[1], bottomright[0], bottomleft[0])
            
            left_yintersect = bottom_slope*left_vertical+bottom_slope_c
            right_xintersect = (bottom_horizontal-right_slope_c) /right_slope
            polygon = [l_m,[left_vertical, left_yintersect], b_m,[b_m[0]+100,b_m[1]], [right_xintersect, bottom_horizontal], r_m]
        
        if angle == 'rear-right':
            mbottomleft, mtopleft, mtopright, mbottomright = min_bbox[0], min_bbox[1], min_bbox[2], min_bbox[3]
            polygon = [l_m, bbox[0], b_m,[b_m[0]+100,b_m[1]],[r_m[0], (b_m[1]+r_m[1])/2], r_m]
                    
    
    return polygon

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle")
    args = parser.parse_args()
    
    angle = args.angle
    img_path = 'rearright1_mask.png'
    dir_path = '/home/ubuntu/Users/maixueqiao/image-processing/'
    
    img, bbox, min_bbox, pst = get_pts(img_path,dir_path)

    # unpack points
    l_m, r_m, t_m, b_m = pst
    bottomleft, topleft, topright, bottomright = bbox[0], bbox[1], bbox[2], bbox[3]
    
    polygon = get_polygon(min_bbox, bbox, pst, angle, bottomright[1], bottomright[0], bottomleft[0])
    obj = Object(polygon)
    smoothed_obj = obj.Smooth_by_Chaikin(number_of_refinements = 2)
    x1 = [i for i,j in smoothed_obj]
    y1 = [j for i,j in smoothed_obj]

    foreground_pil = Image.open('/home/ubuntu/Users/maixueqiao/image-processing/{}_foreground.png'.format(img_path.split('_')[0])).convert('RGBA')

    mask_shadow = Image.new('RGBA', foreground_pil.size)
    pdraw = ImageDraw.Draw(mask_shadow)
    pdraw.polygon([(x1,y1) for x1,y1 in zip(x1,y1)],
                  fill=(0,0,0,150),outline=(0,0,0,25))
    im_mask = mask_shadow.filter(ImageFilter.GaussianBlur(25))
    im_mask.paste(foreground_pil, (0,10),foreground_pil.convert('RGBA'))
    im_mask.save('res.png')