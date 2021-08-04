import cv2
import numpy as np


class RegionLaneDetectionwithOpencv:
    def __init__(self, img_w=1280, img_h=720):
        self.pre_averaged_lines = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]

        self.points_region_1 = np.array([(img_w*0.10, img_h*0.85), (img_w*0.10, img_h*0.60), (img_w*0.45,  img_h*0.55)], 'int32')
        self.points_region_2 = np.array([(img_w*0.15, img_h*0.85), (img_w*0.50, img_h*0.55), (img_w*0.55,  img_h*0.55), (img_w*0.45, img_h*0.85)], 'int32')
        self.points_region_3 = np.array([(img_w*0.55, img_h*0.85), (img_w*0.45, img_h*0.55), (img_w*0.50,  img_h*0.55), (img_w*0.9,  img_h*0.85)], 'int32')
        self.points_region_4 = np.array([(img_w-1,    img_h*0.85), (img_w*0.50, img_h*0.65), (img_w-1,     img_h*0.65)], 'int32')

    def detect(self, image):
        grey  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # larger kernel size to remove the unwanted traffic markings
        gaus  = cv2.GaussianBlur(grey, (5, 5), 0)
        edges = cv2.Canny(gaus,30,150)

        left_line,  left_roi  = self.region_lane_detection(edges, self.points_region_2, 2)
        right_line, right_roi = self.region_lane_detection(edges, self.points_region_3, 3)
        black_lines_1         = self.display_lines(image, np.array([left_line, right_line]), (255,0,0))

        left_line,  left_roi  = self.region_lane_detection(edges, self.points_region_1, 1)
        right_line, right_roi = self.region_lane_detection(edges, self.points_region_4, 4)
        black_lines_2         = self.display_lines(image, np.array([left_line, right_line]), (0,0,255))

        return black_lines_1, black_lines_2


    def region_lane_detection(self, canny_edge_img, points, region_id):
        isolated = self.region(canny_edge_img, points)
        #cv2.imshow("frame", isolated)
    
        if region_id == 1:
            #print("---------- Region 1 ----------")
            lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=20, maxLineGap=1000)
            averaged_lines = self.make_left_line(canny_edge_img, lines, points[0][1])
            if averaged_lines[0] == 0 and averaged_lines[1] == 0 and averaged_lines[2] == 0 and averaged_lines[3] == 0:
                averaged_lines = self.pre_averaged_lines[0]
            else:
                self.pre_averaged_lines[0] = averaged_lines
    
        elif region_id == 2:
            #print("---------- Region 2 ---------")
            lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=1000)
            #averaged_lines = make_left_line(canny_edge_img, lines, canny_edge_img.shape[0])
            averaged_lines = self.make_left_line(canny_edge_img, lines, points[0][1])
            if averaged_lines[0] == 0 and averaged_lines[1] == 0 and averaged_lines[2] == 0 and averaged_lines[3] == 0:
                averaged_lines = self.pre_averaged_lines[1]
            else:
                self.pre_averaged_lines[1] = averaged_lines
    
        elif region_id == 3:
            #print("---------- Region 3 ---------")
            lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=1000)
            #averaged_lines = make_right_line(canny_edge_img, lines, canny_edge_img.shape[0])
            averaged_lines = self.make_right_line(canny_edge_img, lines, points[0][1])
            if averaged_lines[0] == 0 and averaged_lines[1] == 0 and averaged_lines[2] == 0 and averaged_lines[3] == 0:
                averaged_lines = self.pre_averaged_lines[2]
            else:
                self.pre_averaged_lines[2] = averaged_lines
    
        elif region_id == 4:
            #print("---------- Region 4 ----------")
            lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 200, np.array([]), minLineLength=40, maxLineGap=1000)
            averaged_lines = self.make_right_line(canny_edge_img, lines, points[0][1])
            if averaged_lines[0] == 0 and averaged_lines[1] == 0 and averaged_lines[2] == 0 and averaged_lines[3] == 0:
                averaged_lines = self.pre_averaged_lines[3]
            else:
                self.pre_averaged_lines[3] = averaged_lines
    
    
        #print("    averaged_lines", averaged_lines)
    
        return averaged_lines, isolated

    def region(self, image, points):
        height, width = image.shape
        trapezium = np.array([points], 'int32')
        #create a black image with the same dimensions as original image
        mask = np.zeros_like(image)
        #create a mask (triangle that isolates the region of interest in our image)
        #mask = cv2.fillPoly(mask, triangle, 255)
        mask = cv2.fillPoly(mask, trapezium, 255)
        mask = cv2.bitwise_and(image, mask)
        return mask


    
    
    def display_lines(self, image, lines, color):
        #OpenCV is BGR
        lines_image = np.zeros_like(image)
        #make sure array isn't empty
        if lines is not None:
            #for line in lines:
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line
                #draw lines on a black image
                #print("    x1:{}  y1:{}  x2:{}  y2:{}".format(x1,y1,x2,y2))
                if np.isinf(x1) or np.isinf(y1) or np.isinf(x2) or np.isinf(y2):
                    continue
    
                cv2.line(lines_image, (x1, y1), (x2, y2), color, 10)
        
        return lines_image
    
    def make_left_line(self, image, lines, bottom_y):
        left = []
    
        if lines is not None:
            #print("    lines:")
            for line in lines:
                #print("        ", line)
                x1, y1, x2, y2 = line.reshape(4)
                #fit line to points, return slope and y-int
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                slope = parameters[0]
                y_int = parameters[1]
                #lines on the right have positive slope, and lines on the left have neg slope
                if slope < -0.1 and slope >-bottom_y:
                    left.append((slope, y_int))
     
        #takes average among all the columns (column0: slope, column1: y_int)
        left_avg  = np.average(left, axis=0)  if len(left)  !=0 else [0,0]
    
        #print("    left:  ", left)
        #print("     avg: {}".format(left_avg))
        #create lines based on averages calculates
    
        return self.make_points(image, left_avg, bottom_y)
    
    def make_right_line(self, image, lines, bottom_y):
        right = []
    
        if lines is not None:
            #print("    lines:")
            for line in lines:
                #print("        ", line)
                x1, y1, x2, y2 = line.reshape(4)
                #fit line to points, return slope and y-int
                parameters = np.polyfit((x1, x2), (y1, y2), 1)
                slope = parameters[0]
                y_int = parameters[1]
                #lines on the right have positive slope, and lines on the left have neg slope
                if slope > 0.1 and slope < bottom_y:
                    right.append((slope, y_int))
     
        #takes average among all the columns (column0: slope, column1: y_int)
        right_avg  = np.average(right, axis=0)  if len(right)  !=0 else [0,0]
    
        #print("    right:  ", right)
        #print("      avg: {}".format(right_avg))
        #create lines based on averages calculates
    
        return self.make_points(image, right_avg, bottom_y)
    
    
    def make_points(self, image, average, bottom_y):
        slope, y_int = average
    
        if slope == 0 and y_int==0:
            return np.array([0,0,0,0])
    
        #y1 = image.shape[0]
        y1 = bottom_y
        #how long we want our lines to be --> 3/5 the size of the image
        #print("y1: {}   image.shape 0 : {}".format(y1, image.shape[0]))
        #y2 = int(y1 * (5/7)) if y1 == image.shape[0] else int(y1 * (3/5))
        y2 = int(y1 * (2/3))
        #determine algebraically
        x1 = int((y1 - y_int) // slope)
        x2 = int((y2 - y_int) // slope)
    
        y1 = max(min(image.shape[0],y1),0)
        y2 = max(min(image.shape[0],y2),0)
        x1 = max(min(image.shape[1],x1),0)
        x2 = max(min(image.shape[1],x2),0)
    
    
        return np.array([x1, y1, x2, y2])
    
    



