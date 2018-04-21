import cv2
import numpy as np

minValue = 70
def apply_hist_mask(frame, hist):
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	dst = cv2.calcBackProject([hsv], [0,1], hist, [0,180,0,256], 1)
	disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
	cv2.filter2D(dst, -1, disc, dst)
	# ret, thresh = cv2.threshold(dst, 127, 255, 0)
	thresh = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
	thresh = cv2.merge((thresh,thresh, thresh))
	cv2.GaussianBlur(dst, (3,3), 0, dst)
	res = cv2.bitwise_and(frame, thresh)
	return res

def contours(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(gray, 0, 255, 0)
	_ ,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)	
	return contours

def max_contour(contours):
	max_i = 0
	max_area = 0
	
	for i in range(len(contours)):
		cnt = contours[i]
		area = cv2.contourArea(cnt)
		if area > max_area:
			max_area = area
			max_i = i

	contour = contours[max_i]
	return contour

def approxConvexHull(cnt):
	epsilon = 0.01 * cv2.arcLength(cnt, True)
	approx = cv2.approxPolyDP(cnt, epsilon, True)
	return approx

def hull(contour):
	hull = cv2.convexHull(contour, returnPoints=True)
	return hull

def defects(contour):
	hull = cv2.convexHull(contour, returnPoints=False)
	if hull is not None and len(hull > 3) and len(contour) > 3:
		defects = cv2.convexityDefects(contour, hull)	
		return defects
	else: 
		return None

def centroid(contour):
	moments = cv2.moments(contour)
	if moments['m00'] != 0:
		cx = int(moments['m10']/moments['m00'])
		cy = int(moments['m01']/moments['m00'])
		return (cx,cy)
	else:
		return None		

def contour_interior(frame, contour):
	rect = cv2.minAreaRect(contour)
	box = cv2.cv.BoxPoints(rect)
	box = np.int0(box)

	rows,cols,_ = frame.shape
	mask = np.zeros((rows,cols), dtype=np.float)
	for i in xrange(rows):
		for j in xrange(cols):
			mask.itemset((i,j), cv2.pointPolygonTest(box, (j,i), False))

	mask = np.int0(mask)
	mask[mask < 0] = 0
	mask[mask > 0] = 255
	mask = np.array(mask, dtype=frame.dtype)
	mask = cv2.merge((mask, mask, mask))
	
	contour_interior = cv2.bitwise_and(frame, mask)
	return contour_interior			

def gray_threshold(frame, threshold_value):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray, threshold_value, 255, 0)
	return thresh

def farthest_point(defects, contour, centroid):
	s = defects[:,0][:,0]
	cx, cy = centroid
	
	x = np.array(contour[s][:,0][:,0], dtype=np.float)
	y = np.array(contour[s][:,0][:,1], dtype=np.float)
				
	xp = cv2.pow(cv2.subtract(x, cx), 2)
	yp = cv2.pow(cv2.subtract(y, cy), 2)
	dist = cv2.sqrt(cv2.add(xp, yp))

	dist_max_i = np.argmax(dist)

	if dist_max_i < len(s):
		farthest_defect = s[dist_max_i]
		farthest_point = tuple(contour[farthest_defect][0])
		return farthest_point
	else:
		return None	

def plot_farthest_point(frame, point):
    cv2.circle(frame, point, 5, [0,0,255], -1)
def plot_centroid(frame, point):
    cv2.circle(frame, point, 5, [255,0,0], -1)


def plot_hull(frame, hull):
    cv2.drawContours(frame, [hull], 0, (255,0,0), 2)	

def plot_contours(frame, contours):
    cv2.drawContours(frame, contours, -1, (0,255,0), 3)	

def plot_defects(frame, defects, contour):
    if len(defects) > 0:
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])               
            cv2.circle(frame, start, 5, [255,0,255], -1)
def binaryMode(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return res



def plot_points_from_defects(frame,nearsetDict):
	for k in nearsetDict:
		cv2.circle(frame, tuple(nearsetDict[k]), 5, [67,70,231], -1)
		


def findDiff(pt1 , pt2):
	return np.linalg.norm(pt1 - pt2)


def	findNearstPoint(defects, contour):
	max_dist = 30
	groups = {}
	# myList = np.array()
	id = 0
	s0, e0, f0, d0 = defects[0, 0]
	groups['0'] = np.array([contour[s0][0]])
	for i in range(1, defects.shape[0]):
		s, e, f, d = defects[i - 1, 0]
		s2, e2, f2, d2 = defects[i, 0]
		if findDiff(contour[s][0], contour[s2][0]) < max_dist:
			if str(id) in groups:
				groups[str(id)] = np.append(groups[str(id)], np.array([contour[s2][0]]) , axis=0)
			else: 
				groups[str(id)] = np.array([contour[s2][0]])	
		else:
			id += 1
	# get the mean of the points in each group
	nearsetMean = {}
	for k in groups:
		neasetMeanCalc = np.mean(groups[k], axis=0)
		nearsetMean[k] = neasetMeanCalc.astype(int)
	return nearsetMean
