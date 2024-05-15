from itertools import combinations
import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, DPTForDepthEstimation
import torch

#직선의 방정식 찾기
def find_line_equation(circle1, circle2):
    """
    두 원의 중심을 지나는 직선의 방정식
    circle --> (x,y)
    return m,b
    """
    x1, y1 = circle1
    x2, y2 = circle2
    x = [x1,x2]
    y = [y1,y2]
    m,b = np.polyfit(x,y,1)

    return m, b

def calculate_distance(point1, point2):
    """
    두점의 거리
    point --> (x,y)
    return 거리
    """
    x1, y1 = point1
    x2, y2 = point2

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def find_circle_line_intersections(circle, slope, intercept):
    """
    직선의 방정식과 원의 교점을 찾는 함수
    circle --> (x,y,r)
    """
    x_circle = circle[0]
    y_circle = circle[1]
    radius = circle[2]
    a = 1 + slope**2
    b = -2 * x_circle + 2 * slope * (intercept - y_circle)
    c = x_circle**2 + (intercept - y_circle)**2 - radius**2

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return []

    x1 = (-b + math.sqrt(discriminant)) / (2 * a)
    y1 = slope * x1 + intercept

    x2 = (-b - math.sqrt(discriminant)) / (2 * a)
    y2 = slope * x2 + intercept
    
    intersections = [(x1, y1), (x2, y2)]
    
    if not intersections:
        return None

    return intersections

def find_farthest_point(reference_point, point_list):
    """기준점으로부터 가장 먼 점의 좌표를 반환하는 함수"""
    farthest_point = None
    max_distance = 0

    for point in point_list:
        distance = calculate_distance(reference_point, point)
        if distance > max_distance:
            max_distance = distance
            farthest_point = point

    return farthest_point

def find_closest_point(reference_point, point_list):
    """기준점으로부터 가장 가까운 점의 좌표를 반환하는 함수"""
    closest_point = None
    min_distance = float('inf')

    for point in point_list:
        distance = calculate_distance(reference_point, point)
        if distance < min_distance:
            min_distance = distance
            closest_point = point

    return closest_point

def find_vp(xy, m, b, distance,reference_point):
    """
    주어진 기준 좌표 [x, y]로부터 주어진 거리 L에 있는 직선상의 두 점을 찾는 함수.
    직선의 방정식: y = mx + b
    reference_point --> (x,y)
    """
    x = xy[0]
    y = xy[1]
    # 주어진 거리를 이용하여 새로운 좌표를 계산
    delta_x = distance / (1 + m**2)**0.5
    delta_y = m * delta_x

    # 두 점의 좌표를 계산
    point1 = [x + delta_x, y + delta_y]
    point2 = [x - delta_x, y - delta_y]
    point = find_closest_point(reference_point=reference_point,point_list=[point1,point2])
    return point

def vp_method(combo):
    """
    두원에의한 소실점 좌표 계산 함수
    combo_circle --> [[x1,y1,r1],[x2,y2,r2]] done, r1<<r2
    """
    if combo[0][2] < combo[1][2]:
        O_1 = combo[1]#큰원
        O_2 = combo[0]#작은원
    else:
        O_1 = combo[0]#큰원
        O_2 = combo[1]#작은원
    r_1 = O_1[2]
    r_2 = O_2[2]


    cx = [O_1[0], O_2[0]]
    cy = [O_1[1], O_2[1]]
    radii = [r_1, r_2]
    L2 = ((cx[0]-cx[1])**2)+((cy[0]-cy[1])**2)
    L = np.sqrt(L2)
    if L>radii[1]:
        dR_max = (radii[0])-(radii[1]-L)
        dR_min = (radii[0])-(radii[1]+L)
    else:
        dR_max = (radii[0])-radii[1]+L
        dR_min = (radii[0])-radii[1]-L

    OO = calculate_distance(point1=combo[0][:-1],point2=combo[1][:-1])


    L = (dR_max/(dR_max+dR_min))*2*combo[1][2]

    #직선
    m, b = find_line_equation(combo[0][:-1], combo[1][:-1])

    intersection_points = find_circle_line_intersections(combo[1],m,b)
    #print(intersection_points)
    farthest_point = find_farthest_point(combo[0][:-1], intersection_points)
    vp = find_vp(farthest_point,m,b,L,reference_point=O_2[:-1])

    return vp

def dinov2_depth(img_path,img_size,norm=True,revers=True):
    image = Image.open(img_path)
    image = image.resize(img_size)
    image_processor = AutoImageProcessor.from_pretrained("facebook/dpt-dinov2-giant-nyu")
    model = DPTForDepthEstimation.from_pretrained("facebook/dpt-dinov2-giant-nyu")
    # prepare image for the model
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()

    if norm==True:
        depth_map = cv2.normalize((output * 255 / np.max(output)).astype("uint8"),None,0,255,cv2.NORM_MINMAX)
    else:
        depth_map = cv2.normalize((output * 255 / np.max(output)).astype("uint8"),None,0,255,cv2.NORM_MINMAX)

    if revers==True:
        depth_map = 255-depth_map

    else:
        depth_map = depth_map
    
    return depth_map

def circle_method(combo):
    """
    두원에의한 소실점 좌표 계산 함수
    combo --> [[x1,y1,r1],[x2,y2,r2],[m,b]] done, r1<<r2
    """
    if combo[0][2] < combo[1][2]:
        O_1 = combo[1]#큰원
        O_2 = combo[0]#작은원
    else:
        O_1 = combo[0]#큰원
        O_2 = combo[1]#작은원
    r_1 = O_1[2]
    r_2 = O_2[2]


    cx = [O_1[0], O_2[0]]
    cy = [O_1[1], O_2[1]]
    radii = [r_1, r_2]
    L2 = ((cx[0]-cx[1])**2)+((cy[0]-cy[1])**2)
    L = np.sqrt(L2)
    if L>radii[1]:
        dR_max = (radii[0])-(radii[1]-L)
        dR_min = (radii[0])-(radii[1]+L)
    else:
        dR_max = (radii[0])-radii[1]+L
        dR_min = (radii[0])-radii[1]-L

    OO = calculate_distance(point1=combo[0][:-1],point2=combo[1][:-1])


    L = (dR_max/(dR_max+dR_min))*2*combo[1][2]
    dR = dR_max/dR_min
    #직선
    #print(combo[2])
    m, b = combo[2][0], combo[2][1]

    intersection_points = find_circle_line_intersections(combo[1],m,b)
    #print(intersection_points)
    farthest_point = find_closest_point(combo[0][:-1], intersection_points)
    vp = find_vp(farthest_point,m,b,L,reference_point=O_2[:-1])

    return vp