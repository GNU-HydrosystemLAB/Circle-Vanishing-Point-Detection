import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from skimage import feature, transform, io
from itertools import product
from itertools import permutations
from itertools import combinations
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
from PIL import Image
import vp_utill


plt.rcParams['font.family'] = 'Times New Roman'


class Vanishing_Point:
    def __init__(self, image_path, dist, mtx):
        self.master_bin = 10
        self.image_path = image_path
        self.real_pipe_D = 83.6
        self.unit = ['mm','mm2']
        self.f = 1077.02
        self.dist = dist
        self.mtx = mtx
        self.image = cv2.imread(image_path)
        self.undist_image = cv2.undistort(self.image, self.mtx, self.dist)
        self.RgB_un_image = cv2.cvtColor(self.undist_image, cv2.COLOR_BGR2RGB)
        self.G_image = cv2.cvtColor(self.undist_image, cv2.COLOR_BGR2HSV)[:,:,2]
        self.circles_S = []
        self.circles_B = []
        self.slopes = []
        self.Thers_range = 4 #1
        self.master_Hug = 2 #4
        self.Thers_list_S = list(range(1,128,self.Thers_range))#48,96
        self.Thers_list_B = list(range(128,255,self.Thers_range))#96,141
        self.circles_ALL = None
        #----------
        self.slope_filter_circles_S = None
        self.slope_filter_circles_B = None
        #----------
        self.dR_filter_circles = None
        #----------
        self.best_circles_S = None
        self.best_circles_B = None
        self.v_p = None
        self.dR = None
        self.m = None
        self.b = None
        self.c_list = []
        self.visual_c_list = []
        #----------
        self.model = None
        self.mask = None
        self.mask_m_c = None
        self.interest_length_circles = []
        self.interest_width_coeffs = []
        

        
    # 1) 이진화 범위에 따른 원검출과 원 리스트 정의
    def image_processing(self):
        nomal_image = cv2.normalize(self.G_image,None,0,255, cv2.NORM_MINMAX)
        
        for i in self.Thers_list_S:
            self.image_thres(frame=nomal_image, thresh_no=i, append_list="S")

        for i in self.Thers_list_B:
            self.image_thres(frame=nomal_image, thresh_no=i, append_list="B") 

        self.circles_ALL = self.circles_S + self.circles_B
        
    # 2) 1)-->에 쓰이는 허프변환 알고리즘
    def image_thres(self, frame, thresh_no, append_list):
        _, thres = cv2.threshold(cv2.GaussianBlur(frame, (9, 9), 1), thresh_no, 255, cv2.THRESH_BINARY)
        image_circles = cv2.HoughCircles(thres, cv2.HOUGH_GRADIENT, self.master_Hug, 200, param1=250, param2=10, minRadius=0, maxRadius=0)
        if image_circles is not None:
            circle = np.round(image_circles[0, 0]).astype("int")
            
            if append_list == 'S':
                if not any(np.array_equal(circle, existing_circle) for existing_circle in self.circles_S):
                    self.circles_S.append(circle)
            elif append_list == 'B':
                if not any(np.array_equal(circle, existing_circle) for existing_circle in self.circles_B):
                    self.circles_B.append(circle)


    # 3) 원 리스트를 drow_frame에 그리기
    def drow_circles(self,drow_frame,circles_list,color):
        for i in circles_list:
            cv2.circle(drow_frame,(i[0],i[1]),i[2],color,2)
            cv2.circle(drow_frame,(i[0],i[1]),5, color,-1)

    def drow_vp(self, drow_frame,color=(0,0,255)):
        cv2.circle(drow_frame,(self.v_p[0], self.v_p[1]),5, color,-1)
        cv2.line(drow_frame,(self.v_p[0], self.v_p[1]),
                 (self.best_circles_B[0], self.best_circles_B[1]),
                 color,3,cv2.LINE_AA)
    
    # 4) 검출된 원 리스트의 기울기를 계산
    def calculate_slopes(self):
        for i in range(len(self.circles_S)):
            x1, y1, _ = self.circles_S[i]  # S 타입 원의 중심 좌표와 반지름 정보
            for j in range(len(self.circles_B)):
                x2, y2, _ = self.circles_B[j]  # B 타입 원의 중심 좌표와 반지름 정보

                # 기울기 계산
                if x1 != x2:  # 두 원의 x 좌표가 같지 않을 때만 계산 (기울기가 무한대가 되는 경우 방지)
                    slope = (y2 - y1) / (x2 - x1)
                    self.slopes.append([i, j, slope])

        
    def visualize_H(self,Target_list,v='1D'):
        if v=='2D':
            plt.hist(Target_list, bins=self.master_bin, color='blue', alpha=0.7)
            plt.xlabel('x')
            plt.ylabel('Frequency')
            plt.title('Distribution of Slopes')
            plt.grid(True)
            plt.show()    
        elif v=='1D':
            plt.scatter(Target_list, np.zeros(len(Target_list)), c='blue', marker='o', s=10)
            plt.xlabel('Slope')
            plt.title('1D Scatter Plot of Slopes')
            plt.grid(True)
            plt.show()   
                
    # 5) 기울기값의 통계치를 보여줌
    def visualize_slopes(self):
        slopes = [slope for _, _, slope in self.slopes]  # 기울기만 추출
        plt.hist(slopes, bins=self.master_bin, color='blue', alpha=0.7)
        plt.xlabel('Slope')
        plt.ylabel('Frequency')
        plt.title('Distribution of Slopes')
        plt.grid(True)
        plt.show()
        
    def visualize_slopes_scatter(self):
        slopes = [slope for _, _, slope in self.slopes]  # 기울기만 추출
        plt.scatter(slopes, np.zeros(len(slopes)), c='blue', marker='o', s=10)
        plt.xlabel('Slope')
        plt.title('1D Scatter Plot of Slopes')
        plt.grid(True)
        plt.show()
        

    
    def select_dense_slope_range(self, num_ranges=1):
        # 히스토그램을 사용하여 밀집도가 큰 구간 선택
        hist, bin_edges = np.histogram([slope for _, _, slope in self.slopes], bins=self.master_bin)
        dense_ranges = []
        max_density = max(hist)
        for i, density in enumerate(hist):
            if density >= max_density / num_ranges:
                dense_ranges.append((bin_edges[i], bin_edges[i + 1]))

        selected_slopes = []  # 선택된 슬로프 정보만 저장할 리스트

        # 선택된 밀집도 구간 내에서 슬로프 정보 추출
        for start, end in dense_ranges:
            selected_slopes.extend([[idx1, idx2, s] for [idx1, idx2, s] in self.slopes if start <= s <= end])
            
        max_S = []
        max_B = []
        processed_slopes = set()  # 중복된 슬로프 정보를 처리하기 위한 세트

        for i in selected_slopes:
            idx1, idx2, slope = i
            if idx1 not in processed_slopes and idx2 not in processed_slopes:
                max_S.append(self.circles_S[idx1])
                max_B.append(self.circles_B[idx2])
                processed_slopes.add(idx1)  # 중복 처리된 슬로프 정보를 세트에 추가
                processed_slopes.add(idx2)
        self.slope_filter_circles_S = max_S
        self.slope_filter_circles_B = max_B

    def cal_mb(self,x1,y1,x2,y2):
        m = (y2-y1)/(x2-x1)
        b = (x2*y1 - x1*y2)/(x2-x1)
        return m, b
    
    def vanishing_point(self):
        circle1 = self.best_circles_S
        circle2 = self.best_circles_B
        x_set = [circle1[0], circle2[0]] #[x1, x2]
        y_set = [circle1[1], circle2[1]] #[y1, y2]
        r_set = [circle1[2], circle2[2]]
        L = np.sqrt(((x_set[0] - x_set[1])**2) + ((y_set[0] - y_set[1])**2))
        self.m, self.b = self.cal_mb(x_set[0],y_set[0],x_set[1],y_set[1])
        self.dR_max = (r_set[0]) - (r_set[1] - L)
        self.dR_min = (r_set[0]) - (r_set[1] + L)
        self.dR = self.dR_max / self.dR_min
        S_x_to_van_point_L = ((r_set[1] * (self.dR - 1)) - self.dR + 1) / (1 + self.dR)
        A = 1 + self.m**2
        B = 2 * (self.m * self.b - x_set[1] - self.m * y_set[1])
        C = x_set[1]**2 + (self.b - y_set[1])**2 - (S_x_to_van_point_L**2)
        x_vals = np.roots([A, B, C])
        y_vals = self.m * x_vals + self.b
        # 작은 원에서 가까운 점을 소실점으로 선택
        distance_to_first_point = np.sqrt((x_set[0] - x_vals[0])**2 + (y_set[0] - y_vals[0])**2)
        distance_to_second_point = np.sqrt((x_set[0] - x_vals[1])**2 + (y_set[0] - y_vals[1])**2)

        # cx[1], cy[1]에서 더 가까운 거리에 있는 점을 찾기
        if distance_to_first_point < distance_to_second_point:
            vanishing_point_x = x_vals[0]
            vanishing_point_y = y_vals[0]
        else:
            vanishing_point_x = x_vals[1]
            vanishing_point_y = y_vals[1]

        self.v_p = [vanishing_point_x,vanishing_point_y]
        self.v_p = np.round(self.v_p).astype("int")

        
    def create_circle(self,max_r=800):
        c_list = [[self.v_p[0], self.v_p[1], 1]]

        for i in range(700):
            # 현재 원의 반지름과 중심 좌표
            r0 = c_list[-1][2]
            x0 = c_list[-1][0]

            # 기준점
            point = self.best_circles_B[:-1]

            # 반지름 차이
            radius_diff = 1

            # 직선의 방정식
            # 여기에서 m와 b를 적절히 설정하세요

            # L 값을 이용하여 x1 계산
            L = (self.dR - 1) / (self.dR + 1)

            # 두 개의 후보 값 계산
            x1_candidate_1 = x0 + L / np.sqrt(1 + self.m**2)
            x1_candidate_2 = x0 - L / np.sqrt(1 + self.m**2)

            # 기준점으로부터 각 후보에 대한 거리 계산
            distance_1 = np.sqrt((x1_candidate_1 - point[0])**2)
            distance_2 = np.sqrt((x1_candidate_2 - point[0])**2)

            if r0 < self.best_circles_B[2]:
                # 기준점으로부터 더 가까운 값을 선택
                if distance_1 < distance_2:
                    x1 = x1_candidate_1
                else:
                    x1 = x1_candidate_2


            else:
                # 기준점으로부터 더 가까운 값을 선택
                if distance_1 < distance_2:
                    x1 = x1_candidate_2
                else:
                    x1 = x1_candidate_1
            
            # 다음 원의 반지름
            y1 = self.m * x1 + self.b
            r1 = r0 + radius_diff

            # 원을 추가
            c_list.append([x1, y1, r1])
            
        self.c_list = c_list

    
    def calculate_dR(self):
        dR_list = []
        for i in self.slope_filter_circles_S:
            for j in self.slope_filter_circles_B:
                circle1 = i
                circle2 = j
                x_set = [circle1[0], circle2[0]]
                y_set = [circle1[1], circle2[1]]
                r_set = [circle1[2], circle2[2]]
                L = np.sqrt(((x_set[0] - x_set[1])**2) + ((y_set[0] - y_set[1])**2))
                m, b = self.cal_mb(x_set[0],y_set[0],x_set[1],y_set[1])
                dR_max = (r_set[0]) - (r_set[1] - L)
                dR_min = (r_set[0]) - (r_set[1] + L)
                dR = dR_max / dR_min
                append_list = [i,j,dR]
                dR_list.append(append_list)
                
        self.dR_filter_circles = dR_list
        
        
    def visualize_dR(self, v="1D"):
        dR = []
        for i in self.dR_filter_circles:
            dR.append(i[2])
        if v == "2D":
            plt.hist(dR, bins=self.master_bin, color='blue', alpha=0.7)
            plt.xlabel('Slope')
            plt.ylabel('Frequency')
            plt.title('Distribution of dR')
            plt.grid(True)
            plt.show()
        
        elif v == '1D':
            plt.scatter(dR, np.zeros(len(dR)), c='blue', marker='o', s=10)
            plt.xlabel('Slope')
            plt.title('1D Scatter Plot of dR')
            plt.grid(True)
            plt.show()
            
    def select_dense_dR_range(self, num_ranges=1):
        dR = [item[2] for item in self.dR_filter_circles]
        # 히스토그램을 사용하여 밀집도가 큰 구간 선택
        hist, bin_edges = np.histogram(dR, bins=self.master_bin)
        dense_ranges = []
        max_density = max(hist)
        for i, density in enumerate(hist):
            if density >= max_density / num_ranges:
                dense_ranges.append((bin_edges[i], bin_edges[i + 1]))
        
        # 밀집도 구간의 중앙값 계산
        middle_values = [(start + end) / 2 for start, end in dense_ranges]

        # 중앙값과 가장 가까운 dR 값을 가진 리스트 전체 찾기
        closest_dR_values = []

        for middle in middle_values:
            # 중앙값과 가장 가까운 dR 값을 찾기
            min_distance = float('inf')
            closest_dR_item = None

            for item in self.dR_filter_circles:
                dR_value = item[2]
                distance = abs(dR_value - middle)

                if distance < min_distance:
                    min_distance = distance
                    closest_dR_item = item

            # 가장 가까운 dR 값을 가진 아이템을 결과 리스트에 추가
            closest_dR_values.append(closest_dR_item)
        return closest_dR_values
    
    def select_dense_dR_mean_range(self, num_ranges=1):
        dR = [item[2] for item in self.dR_filter_circles]
        # 히스토그램을 사용하여 밀집도가 큰 구간 선택
        hist, bin_edges = np.histogram(dR, bins=self.master_bin)
        dense_ranges = []
        max_density = max(hist)
        for i, density in enumerate(hist):
            if density >= max_density / num_ranges:
                dense_ranges.append((bin_edges[i], bin_edges[i + 1]))

        # 밀집도 구간 내에 있는 모든 dR 값을 가져오기
        all_dR_values = []

        for start, end in dense_ranges:
            dR_values_in_range = [item[2] for item in self.dR_filter_circles if start <= item[2] <= end]
            all_dR_values.extend(dR_values_in_range)

        # dR 값들의 평균값 계산
        mean_dR = np.mean(all_dR_values)
        
        closest_dR_values = []

            # 중앙값과 가장 가까운 dR 값을 찾기
        min_distance = float('inf')
        closest_dR_item = None

        for item in self.dR_filter_circles:
            dR_value = item[2]
            distance = abs(dR_value - mean_dR)

            if distance < min_distance:
                min_distance = distance
                closest_dR_item = item

            # 가장 가까운 dR 값을 가진 아이템을 결과 리스트에 추가
        closest_dR_values.append(closest_dR_item)
            
        return closest_dR_values
    
    def S_dR_nomal(self, data):
        #정규화
        slope_values = [sub_list[2] for sub_list in data]
        dR_values = [sub_list[3] for sub_list in data]
        # Min-Max 스케일링을 통한 정규화
        min_slope = min(slope_values)
        max_slope = max(slope_values)
        min_dR = min(dR_values)
        max_dR = max(dR_values)

        normalized_slope = [(x - min_slope) / (max_slope - min_slope) for x in slope_values]
        normalized_dR = [(x - min_dR) / (max_dR - min_dR) for x in dR_values]

        # 정규화된 값으로 기존 리스트 업데이트
        for i in range(len(data)):
            data[i][2] = normalized_slope[i]
            data[i][3] = normalized_dR[i]





        slope_mean = np.mean([sub_list[2] for sub_list in data])
        dR_mean = np.mean([sub_list[3] for sub_list in data])

        # 가장 가까운 데이터 찾기
        min_distance = float('inf')  # 초기 최소 거리 설정
        closest_data = None  # 초기 가장 가까운 데이터 설정

        for sub_list in data:
            slope, dR = sub_list[2], sub_list[3]
            distance = np.sqrt(((slope - slope_mean) ** 2) + ((dR - dR_mean) ** 2))  # 2차원 평면에서의 거리 계산
            if distance < min_distance:
                min_distance = distance
                closest_data = sub_list

        # 결과 출력
        return slope_mean, dR_mean, closest_data


    def YOLO_mask(self):
        results = self.model.predict(source=self.undist_image, show=False, conf=0.25,retina_masks=True, boxes=False,save=False)

        masks = results[0].masks
        masks_np = masks.data.cpu().numpy()  # GPU에서 CPU로 이동한 후 numpy 배열로 변환
        num_masks = masks_np.shape[0]  # 마스크 개수

        combined_mask = np.zeros_like(masks_np[0])  # 결합된 마스크를 저장할 배열

        # 각 마스크를 이진 이미지로 변환하여 결합
        for i in range(num_masks):
            mask = masks_np[i]  # i번째 마스크
            combined_mask[mask > 0.5] = 255
        combined_mask = combined_mask.astype(np.uint8)
        self.mask = combined_mask
        
        moments = cv2.moments(combined_mask)
        x = int(moments["m10"] / moments["m00"]) if moments["m00"] != 0 else None
        y = int(moments["m01"] / moments["m00"]) if moments["m00"] != 0 else None

        self.mask_m_c = [x, y]
        #io.imsave('mask.png', self.mask)
        
    def interest_length(self):
        edge = feature.canny(self.mask)
        # Find the circle with the smallest radius containing the most edge pixels
        max_edge_pixels = 0
        selected_circle_max = None

        for center_x, center_y, r in self.c_list:
            # Create a mask for the current circle
            y, x = np.ogrid[-center_y:self.image.shape[0] - center_y, -center_x:self.image.shape[1] - center_x]
            c = x ** 2 + y ** 2 <= r ** 2

            # Calculate the number of edge pixels within the circle
            edge_pixels_in_circle = np.sum(edge & c)

            # Update the selected circle if more edge pixels are found
            if edge_pixels_in_circle > max_edge_pixels:
                max_edge_pixels = edge_pixels_in_circle
                selected_circle_max = [center_x, center_y, r]

        # Extract the selected circle information
        self.interest_length_circles.append(selected_circle_max)

        # Find the circle with the largest radius containing the fewest edge pixels
        min_edge_pixels = float('inf')
        selected_circle_min = None

        for center_x, center_y, r in self.c_list:
            # Create a mask for the current circle
            y, x = np.ogrid[-center_y:self.image.shape[0] - center_y, -center_x:self.image.shape[1] - center_x]
            c = x ** 2 + y ** 2 <= r ** 2

            # Calculate the number of edge pixels within the circle
            edge_pixels_in_circle = np.sum(edge & c)

            # Update the selected circle if fewer edge pixels are found
            if edge_pixels_in_circle < min_edge_pixels and edge_pixels_in_circle > 0:
                min_edge_pixels = edge_pixels_in_circle
                selected_circle_min = [center_x, center_y, r]
                
        self.interest_length_circles.append(selected_circle_min)
        L_max_R = (self.real_pipe_D)*(self.f/(selected_circle_max[2]*2))
        L_min_R = (self.real_pipe_D)*(self.f/(selected_circle_min[2]*2))
        interest_length = L_min_R-L_max_R
        print(f'Area of interest length: {interest_length}',self.unit[0])
        
    
    def matplot_interest_length(self):
        # 시각화
        none_image = np.zeros_like([self.RgB_un_image.shape])
        fig, ax = plt.subplots()
        ax.imshow(self.RgB_un_image)
        
        plt.plot(self.v_p[0], self.v_p[1], 'yo', markersize=3)
        plt.plot(self.mask_m_c[0], self.mask_m_c[1], 'ro', markersize=3)
        plt.text(self.v_p[0], self.v_p[1], "Vanishing Point", color='y', fontsize=8, verticalalignment='bottom')
        plt.text(self.mask_m_c[0], self.mask_m_c[1], "Target", color='r', fontsize=8, verticalalignment='bottom')
        
        plt.plot(self.best_circles_S[0], self.best_circles_S[1], 'b' + 'o', markersize=3)
        circle = plt.Circle((self.best_circles_S[0], self.best_circles_S[1]), self.best_circles_S[2], edgecolor='b', facecolor='none')
        ax.add_patch(circle)
        
        plt.plot(self.best_circles_B[0], self.best_circles_B[1], 'b' + 'o', markersize=3)
        circle = plt.Circle((self.best_circles_B[0], self.best_circles_B[1]), self.best_circles_B[2], edgecolor='b', facecolor='none')
        ax.add_patch(circle)
            
        for i in self.interest_length_circles:
            plt.plot(i[0], i[1], 'r' + 'o', markersize=3)
            circle = plt.Circle((i[0], i[1]), i[2], edgecolor='r', facecolor='none')
            ax.add_patch(circle)
        
        plt.plot([self.v_p[0],self.best_circles_B[0]],
                [self.v_p[1],self.best_circles_B[1]], 'g--', linewidth=2)
        
        for i in self.interest_width_coeffs:
            plt.plot([self.v_p[0],i[1]],
                    [self.v_p[1],i[2]], 'r--', linewidth=2)
        
        plt.show()
        
    def interest_width(self):
        edge = feature.canny(self.mask)
        all_lines = []  # 모든 직선들의 정보를 저장할 리스트
        for edge_pixel in zip(np.nonzero(edge)[1], np.nonzero(edge)[0]):
            x_edge, y_edge = edge_pixel
            coeffs = np.polyfit([x_edge, self.v_p[0]], [y_edge, self.v_p[1]], 1)
            all_lines.append([coeffs,x_edge,y_edge])

        # 기울기 기준으로 정렬하여 가장 기울기가 급한 직선과 완만한 직선 선택
        all_lines.sort(key=lambda line: line[0][0], reverse=True)
        steepest_line_coeffs = all_lines[0]
        gentlest_line_coeffs = all_lines[-1]
        self.interest_width_coeffs.append(steepest_line_coeffs)
        self.interest_width_coeffs.append(gentlest_line_coeffs)
        
    def slope_dR_comb(self):
        comb = list(permutations(self.circles_ALL, 2))
        comb_df = []
 
        for i in comb:
            
            if  i[0][0] != i[1][0]:  # 두 원의 x 좌표가 같지 않을 때만 계산 (기울기가 무한대가 되는 경우 방지)
                slope = (i[1][1] - i[0][1]) / (i[1][0] - i[0][0])


            elif i[0][0] == i[1][0]:
                slope = float('inf')

                
            if i[0][2] < i[1][2]:
                circle1 = i[0]
                circle2 = i[1]
            elif i[0][2] > i[1][2]:
                circle1 = i[1]
                circle2 = i[0]
                
        
            x_set = [circle1[0], circle2[0]]
            y_set = [circle1[1], circle2[1]]
            r_set = [circle1[2], circle2[2]]
            L = np.sqrt(((x_set[0] - x_set[1])**2) + ((y_set[0] - y_set[1])**2))
            m, b = self.cal_mb(x_set[0],y_set[0],x_set[1],y_set[1])
            dR_max = (r_set[0]) - (r_set[1] - L)
            dR_min = (r_set[0]) - (r_set[1] + L)
            dR = dR_max / dR_min
            
            comb_df.append([i[0],i[1],slope,dR])
        
            slope_dR_DF = [sub_list for sub_list in comb_df if sub_list[3] >= 0 and sub_list[2] != float('inf')]
            slopes = [slope[2] for slope in slope_dR_DF]
            dRs = [dR[3] for dR in slope_dR_DF]

        return slope_dR_DF, slopes, dRs
    
    def slope_filter(self,data):
        slopes = [sub_list[2] for sub_list in data]

        # 히스토그램 생성
        hist, bin_edges = np.histogram(slopes, bins=self.master_bin)

        # 가장 높은 밀집도를 갖는 구간을 찾음
        max_density_bin = np.argmax(hist)

        # 가장 높은 밀집도를 갖는 구간의 경계값
        bin_start, bin_end = bin_edges[max_density_bin], bin_edges[max_density_bin + 1]

        # 가장 높은 밀집도를 갖는 구간 내에 있는 데이터만 추출
        filtered_data = [sub_list for sub_list in data if bin_start <= sub_list[2] <= bin_end]
        return filtered_data
    
    def dR_filter(self,data):
        slopes = [sub_list[3] for sub_list in data]

        # 히스토그램 생성
        hist, bin_edges = np.histogram(slopes, bins=self.master_bin)

        # 가장 높은 밀집도를 갖는 구간을 찾음
        max_density_bin = np.argmax(hist)

        # 가장 높은 밀집도를 갖는 구간의 경계값
        bin_start, bin_end = bin_edges[max_density_bin], bin_edges[max_density_bin + 1]

        # 가장 높은 밀집도를 갖는 구간 내에 있는 데이터만 추출
        filtered_data = [sub_list for sub_list in data if bin_start <= sub_list[3] <= bin_end]
        return filtered_data
    
    def vp_filter(self,data):

        x = [sub_list[4] for sub_list in data]

        # 히스토그램 생성
        hist, bin_edges = np.histogram(x, bins=self.master_bin)

        # 가장 높은 밀집도를 갖는 구간을 찾음
        max_density_bin = np.argmax(hist)

        # 가장 높은 밀집도를 갖는 구간의 경계값
        bin_start, bin_end = bin_edges[max_density_bin], bin_edges[max_density_bin + 1]

        # 가장 높은 밀집도를 갖는 구간 내에 있는 데이터만 추출
        x_filtered_data = [sub_list for sub_list in data if bin_start <= sub_list[4] <= bin_end]
        
        
        y = [sub_list[5] for sub_list in x_filtered_data]

        # 히스토그램 생성
        hist, bin_edges = np.histogram(y, bins=self.master_bin)

        # 가장 높은 밀집도를 갖는 구간을 찾음
        max_density_bin = np.argmax(hist)

        # 가장 높은 밀집도를 갖는 구간의 경계값
        bin_start, bin_end = bin_edges[max_density_bin], bin_edges[max_density_bin + 1]

        # 가장 높은 밀집도를 갖는 구간 내에 있는 데이터만 추출
        x_y_filtered_data = [sub_list for sub_list in x_filtered_data if bin_start <= sub_list[5] <= bin_end]
        
        
        
        
        
        return x_y_filtered_data
    
    def S_dR_scatter(self,data, mean_slope_dR=None,sel_slope_dR=None):
        # 데이터에서 slope와 dR 값을 추출
        slopes = [sub_list[2] for sub_list in data]
        dR_values = [sub_list[3] for sub_list in data]

        fig = plt.figure(figsize=(16, 10), dpi=80)
        grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

        # Define the axes
        ax_main = fig.add_subplot(grid[:-1, :-1])
        ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
        ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

        # Scatterplot on main ax
        ax_main.scatter(slopes, dR_values, s=40, alpha=1, c='navy', edgecolors='silver', linewidths=0.2)
        if mean_slope_dR != None:
            ax_main.scatter(mean_slope_dR[0],mean_slope_dR[1] ,s=40, alpha=1, marker='x',c='red', edgecolors='silver', linewidths=1.2)
        if sel_slope_dR !=None:
            ax_main.scatter(sel_slope_dR[2],sel_slope_dR[3] ,s=110, alpha=1, marker='*',c='red', edgecolors='red', linewidths=1.3)
        

        
        # Histogram on the right
        ax_bottom.hist(slopes, 40, histtype='stepfilled', orientation='vertical', color='goldenrod', edgecolor ='white', linewidth = 1)
        ax_bottom.invert_yaxis()

        # Histogram in the bottom
        ax_right.hist(dR_values, 40, histtype='stepfilled', orientation='horizontal', color='goldenrod', edgecolor ='white' ,linewidth = 1)

        # Decorations
        ax_main.set(title='d', xlabel='slope', ylabel='dR')
        ax_main.title.set_fontsize(20)
        for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
            item.set_fontsize(14)

        xlabels = ax_main.get_xticks().tolist()
        ax_main.set_xticklabels(xlabels)
        ax_main.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))  # X 축을 첫째 자리에서 반올림하여 표시
        ax_main.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

        plt.show()
        
    def v_p_add(self, data):
        SdRvP_F_DF = []

        for i in data:
            if i[0][2] < i[1][2]:
                circle1 = i[0]
                circle2 = i[1]
            else:
                circle1 = i[1]
                circle2 = i[0]

            x_set = [circle1[0], circle2[0]]
            y_set = [circle1[1], circle2[1]]
            r_set = [circle1[2], circle2[2]]
            L = np.sqrt(((x_set[0] - x_set[1])**2) + ((y_set[0] - y_set[1])**2))
            m, b = self.cal_mb(x_set[0], y_set[0], x_set[1], y_set[1])
            dR_max = (r_set[0]) - (r_set[1] - L)
            dR_min = (r_set[0]) - (r_set[1] + L)

            # Check if dR_min is zero and handle it
            if dR_min == 0:
                
                continue

            dR = dR_max / dR_min
            S_x_to_van_point_L = ((r_set[1] * (dR - 1)) - dR + 1) / (1 + dR)
            A = 1 + m**2
            B = 2 * (m * b - x_set[1] - m * y_set[1])
            C = x_set[1]**2 + (b - y_set[1])**2 - (S_x_to_van_point_L**2)

            try:
                x_vals = np.roots([A, B, C])
                y_vals = m * x_vals + b
            except np.linalg.LinAlgError:
                
                continue

            distance_to_first_point = np.sqrt((x_set[0] - x_vals[0])**2 + (y_set[0] - y_vals[0])**2)
            distance_to_second_point = np.sqrt((x_set[0] - x_vals[1])**2 + (y_set[0] - y_vals[1])**2)

            if distance_to_first_point < distance_to_second_point:
                vanishing_point_x = x_vals[0]
                vanishing_point_y = y_vals[0]
            else:
                vanishing_point_x = x_vals[1]
                vanishing_point_y = y_vals[1]

            v_p = [vanishing_point_x, vanishing_point_y]
            x, y = round(v_p[0]), round(v_p[1])
            SdRvP_F_DF.append([i[0], i[1], i[2], i[3], x, y])

        
        
        return SdRvP_F_DF
    
    def new_v_p_add(self, data):
        SdRvP_F_DF = []

        for i in data:
            if i[0][2] < i[1][2]:
                circle1 = i[0]
                circle2 = i[1]
            else:
                circle1 = i[1]
                circle2 = i[0]

            vp = vp_utill.vp_method([circle1,circle2])
            new_vp = vp_utill.vp_method([circle1,circle2])

            SdRvP_F_DF.append([i,new_vp])
        
        
        return SdRvP_F_DF

    def vp_scatter(self,data, mean_slope_dR=None,sel_slope_dR=None):
        # 데이터에서 slope와 dR 값을 추출

        slopes = [sub_list[-2] for sub_list in data]
        dR_values = [sub_list[-1] for sub_list in data]

        fig = plt.figure(figsize=(16, 10), dpi=80)
        grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

        # Define the axes
        ax_main = fig.add_subplot(grid[:-1, :-1])
        ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
        ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

        # Scatterplot on main ax
        ax_main.scatter(slopes, dR_values, s=40, alpha=1, c='navy', edgecolors='silver', linewidths=0.2)
        if mean_slope_dR != None:
            ax_main.scatter(mean_slope_dR[0],mean_slope_dR[1] ,s=40, alpha=1, marker='x',c='red', edgecolors='silver', linewidths=1.2)
        if sel_slope_dR !=None:
            ax_main.scatter(sel_slope_dR[2],sel_slope_dR[3] ,s=110, alpha=1, marker='*',c='red', edgecolors='red', linewidths=1.3)
        

        
        # Histogram on the right
        ax_bottom.hist(slopes, 40, histtype='stepfilled', orientation='vertical', color='goldenrod', edgecolor ='white', linewidth = 1)
        ax_bottom.invert_yaxis()

        # Histogram in the bottom
        ax_right.hist(dR_values, 40, histtype='stepfilled', orientation='horizontal', color='goldenrod', edgecolor ='white' ,linewidth = 1)

        # Decorations
        ax_main.set(title='vp', xlabel='x', ylabel='y')
        ax_main.title.set_fontsize(20)
        for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
            item.set_fontsize(14)

        xlabels = ax_main.get_xticks().tolist()
        ax_main.set_xticklabels(xlabels)
        ax_main.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))  # X 축을 첫째 자리에서 반올림하여 표시
        ax_main.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

        plt.show()