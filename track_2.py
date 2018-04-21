import cv2
import numpy as np
import image_analysis
import os
# import pyautogui

path = "Class2"
count = 0
img_width = 128
img_height = 128

def draw_hand_rect(frame):  
    rows,cols,_ = frame.shape
    hand_row_nw = np.array([6*rows/20,6*rows/20,6*rows/20,10*rows/20,10*rows/20,10*rows/20,14*rows/20,14*rows/20,14*rows/20])

    hand_col_nw = np.array([9*cols/20,10*cols/20,11*cols/20,9*cols/20,10*cols/20,11*cols/20,9*cols/20,10*cols/20,11*cols/20])

    hand_row_se = hand_row_nw + 10
    hand_col_se = hand_col_nw + 10
    size = hand_row_nw.size
    for i in range(size):
        cv2.rectangle(frame,(int(hand_col_nw[i]),int(hand_row_nw[i])),(int(hand_col_se[i]),int(hand_row_se[i])),(0,255,0),1)
        black = np.zeros(frame.shape, dtype=frame.dtype)
        frame_final = np.vstack([black, frame])
    return frame_final

def set_hand_hist(frame):  

    rows,cols,_ = frame.shape
    hand_row_nw = np.array([6*rows/20,6*rows/20,6*rows/20,10*rows/20,10*rows/20,10*rows/20,14*rows/20,14*rows/20,14*rows/20])
    hand_col_nw = np.array([9*cols/20,10*cols/20,11*cols/20,9*cols/20,10*cols/20,11*cols/20,9*cols/20,10*cols/20,11*cols/20])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90,10,3], dtype=hsv.dtype)

    size = hand_row_nw.size
    for i in range(size):
        roi[i*10:i*10+10,0:10] = hsv[int(hand_row_nw[i]):int(hand_row_nw[i])+10, int(hand_col_nw[i]):int(hand_col_nw[i])+10]

    hand_hist = cv2.calcHist([roi],[0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)
    return hand_hist

def draw_final(frame, hand_hist):  
    global img_width, img_height, count
    hand_masked = image_analysis.apply_hist_mask(frame,hand_hist)

    contours = image_analysis.contours(hand_masked)
    # image_analysis.plot_contours(frame , contours)
    if contours is not None and len(contours) > 0:
        max_contour = image_analysis.max_contour(contours)
        x, y, w, h = cv2.boundingRect(max_contour)
        hand = frame[y:y+ h, x : x + w]
        cv2.imshow("hand" , hand)
        if cv2.waitKey(1) & 0xFF == ord('i'):
            os.makedirs("images_dataset/{}".format(path), exist_ok = True)
            hand = cv2.resize(hand , (img_width , img_height))
            cv2.imwrite("images_dataset/{}/{}.png".format(path , count), hand)
            count = count + 1 
            print(count)
        approx = image_analysis.approxConvexHull(max_contour)
        # cv2.drawContours(frame, [approx], -1, (0, 255, 0), 1)
        # extLeft = tuple(max_contour[max_contour[:, :, 0].argmin()][0])
        # extRight = tuple(max_contour[max_contour[:, :, 0].argmax()][0])
        # extTop = tuple(max_contour[max_contour[:, :, 1].argmin()][0])
        # extBot = tuple(max_contour[max_contour[:, :, 1].argmax()][0])

        # cv2.circle(frame, extLeft, 8, (0, 0, 255), -1)
        # cv2.circle(frame, extRight, 8, (0, 255, 0), -1)
        # cv2.circle(frame, extTop, 8, (255, 0, 0), -1)
        # cv2.circle(frame, extBot, 8, (255, 255, 0), -1)

        hull = image_analysis.hull(max_contour)
        defects = image_analysis.defects(max_contour)
        nearsetMeanPoints = image_analysis.findNearstPoint(defects , max_contour)
        # print('Nearset : ' , nearsetMeanPoints)
        image_analysis.plot_points_from_defects(frame , nearsetMeanPoints)
        # image_analysis.plot_defects(frame , defects , max_contour)
        centroid = image_analysis.centroid(max_contour)
        # pyautogui.moveTo(centroid)
        image_analysis.plot_centroid(frame , centroid)
        image_analysis.plot_hull(frame , hull)

def resize(frame):
    rows,cols,_ = frame.shape
    
    ratio = float(cols)/float(rows)
    new_rows = 400
    new_cols = int(ratio*new_rows)
    
    row_ratio = float(rows)/float(new_rows)
    col_ratio = float(cols)/float(new_cols)
    
    resized = cv2.resize(frame, (new_cols, new_rows))
    resized = cv2.flip(resized , 1)	
    return resized


cap = cv2.VideoCapture(0)

handHist = None
trained_hand = False
while(cap.isOpened()):
    ret, frame = cap.read()
    orig = frame.copy()
    frame = cv2.flip(frame , 1)
    frame = resize(frame)
    if cv2.waitKey(1) & 0xFF == ord('h'):
        if trained_hand == False:
            handHist = set_hand_hist(frame)
            trained_hand = True
    if trained_hand == False:
        frame_final = draw_hand_rect(frame)
    elif trained_hand == True:
        draw_final(frame , handHist)

    cv2.imshow("frame" , frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()