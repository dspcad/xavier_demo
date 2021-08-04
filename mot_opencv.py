import cv2
import numpy as np


class MOTwithOpencv:
    def __init__(self):
        self.type_of_trackers = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']
        self.multi_tracker = cv2.MultiTracker_create()

    def reset(self):
        self.multi_tracker = cv2.MultiTracker_create()

    def trackerGen(self, tracker_type):
        """
        Create object tracker.
     
        :param type_of_tracker string: OpenCV tracking algorithm
        """
        if tracker_type == 0:
            tracker = cv2.TrackerBoosting_create()
        elif tracker_type == 1:
            tracker = cv2.TrackerMIL_create()
        elif tracker_type == 2:
            tracker = cv2.TrackerKCF_create()
        elif tracker_type == 3:
            tracker = cv2.TrackerTLD_create()
        elif tracker_type == 4:
            tracker = cv2.TrackerMedianFlow_create()
        elif tracker_type == 5:
            tracker = cv2.TrackerMOSSE_create()
        elif tracker_type == 6:
            tracker = cv2.TrackerCSRT_create()
        else:
            tracker = None


        #if tracker == None:
        #    print('The name of the tracker is incorrect')
        #    print('Here are the possible trackers:')
        #    for idx, t in enumerate(self.type_of_trackers):
        #        print("    {}. {}".format(idx, t))
        #else:
        #    print("Tracking Algorithm {} is selected.".format(self.type_of_trackers[tracker_type]))
        return tracker

    def addTracker(self, tracker_type, img, bbox):
        x = bbox[0]
        y = bbox[1]
        w = bbox[2]
        h = bbox[3]
        self.multi_tracker.add(self.trackerGen(tracker_type), img, (x,y,w,h))

    def update(self, img):
        return self.multi_tracker.update(img);
   

def test():
    img = cv2.imread("images/cat_3.jpg")
    print("img: ", img.shape)
    a = MOTwithOpencv()
    a.addTracker(0,img, (1,2,100,321))
    a.addTracker(1,img, (1,2,100,321))
    a.addTracker(2,img, (1,2,100,321))
    a.addTracker(3,img, (1,2,100,321))
    a.addTracker(4,img, (1,2,100,321))
    a.addTracker(5,img, (1,2,100,321))
    a.addTracker(6,img, (1,2,100,321))

if __name__ == "__main__":
    test()
