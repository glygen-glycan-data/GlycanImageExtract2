import cv2, sys,os
import ImageGrab
import numpy as np
from PIL import Image
import time
def countmono(path):
    GlcNAc = "0"  # blue square
    GalNAc = "0"  # yellow square
    NeuAc = "0"  # purple dia
    Man = "0"  # green circle
    Gal = "0"  # yellow circle
    Fuc = "0"  # red tri
    dictionary = {}
    img_file = cv2.imread(path)
    img_file=cv2.resize(img_file,None,fx=2,fy=2)
    final = img_file.copy()
    cv2.imshow("img",img_file)
    cv2.waitKey(0)



    #threhold to remove blur color
    threshold_var =100
    ret, image = cv2.threshold(img_file, 100, 255, cv2.THRESH_BINARY)  # 140
    cv2.imshow(f'threshold: {threshold_var}', image)
    cv2.waitKey(0)

    # masking to remove black

    '''
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    filtered = cv2.filter2D(image, -1, kernel)
    cv2.imshow('filtered', filtered)
    cv2.waitKey(0)
    '''
    grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow('grey', grey)
    cv2.waitKey(0)

    ret, inv_threshold = cv2.threshold(grey, 240, 255, cv2.THRESH_BINARY_INV)  # 140
    cv2.imshow('inv_threshold', inv_threshold)
    cv2.waitKey(0)

    contours_list, hierarchy = cv2.findContours(inv_threshold,
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = []
    count = 0

    for contour in contours_list:
        x, y, w, h = cv2.boundingRect(contour)
        p1 = (x, y)
        p2 = (x + w, y + h)
        print(contour)
        contours.append((p1, p2))
        cv2.rectangle(final, p1, p2, (0, 0, 255), 1)
        cv2.rectangle(grey, p1, p2, (0, 0, 255), 1)
        approx=cv2.approxPolyDP(contour,0.045*cv2.arcLength(contour,True),True)
        cv2.drawContours(final,[approx],0,(0, 0, 255),2)
        print(len(approx))
        if len(approx)==3:
            cv2.putText(final,"Fuc",(approx.ravel()[0],approx.ravel()[1]),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0, 0, 255))
        if len(approx) == 4:
            cv2.putText(final, "rec", (approx.ravel()[0], approx.ravel()[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        (0, 0, 255))
        if len(approx) > 4:
            cv2.putText(final, "cir", (approx.ravel()[0], approx.ravel()[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        (0, 0, 255))

    cv2.imshow('final2', final)
    cv2.waitKey(0)
    cv2.imshow('grey2', grey)
    cv2.waitKey(0)

    return dictionary

def recordoperation(x):
    pass
def capturescreen():
    last_time = time.time()
    cv2.namedWindow("Change HSV limit")

    cv2.createTrackbar("lower H", "Change HSV limit",0,180,recordoperation)
    cv2.createTrackbar("lower S", "Change HSV limit", 0, 255, recordoperation)
    cv2.createTrackbar("lower V", "Change HSV limit", 0, 255, recordoperation)
    cv2.createTrackbar("upper H", "Change HSV limit", 180, 180, recordoperation)
    cv2.createTrackbar("upper S", "Change HSV limit", 255, 255, recordoperation)
    cv2.createTrackbar("upper V", "Change HSV limit", 255, 255, recordoperation)
    cv2.createTrackbar("PolyDP Threshold", "Change HSV limit", 0, 500, recordoperation)
    cv2.createTrackbar("area", "Change HSV limit", 0, 500, recordoperation)

    while (True):
        screen = ImageGrab.grab(bbox=(0, 300, 300, 600))
        screen = np.array(screen.getdata(), dtype = 'uint8').reshape((screen.size[1], screen.size[0], 3))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        screen_final = screen.copy()
        hsv = cv2.cvtColor(screen,cv2.COLOR_BGR2HSV)


        ret, screen = cv2.threshold(screen, 100, 255, cv2.THRESH_BINARY)
        #cv2.imshow("Original", np.array(screen_final))
        grey = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        ret, screen = cv2.threshold(grey, 240, 255, cv2.THRESH_BINARY_INV)  # 140
        print(f'FPS: {1/float(format(time.time() - last_time))} ')

        #screen = cv2.cvtColor(screen,cv2.Color_BGR2RGB)

        # trackbar sliders
        l_h = cv2.getTrackbarPos("lower H", "Change HSV limit")
        l_s = cv2.getTrackbarPos("lower S", "Change HSV limit")
        l_v = cv2.getTrackbarPos("lower V", "Change HSV limit")
        u_h = cv2.getTrackbarPos("upper H", "Change HSV limit")
        u_s = cv2.getTrackbarPos("upper S", "Change HSV limit")
        u_v = cv2.getTrackbarPos("upper V", "Change HSV limit")
        p_t = cv2.getTrackbarPos("PolyDP Threshold", "Change HSV limit")
        cnt_area = cv2.getTrackbarPos("area", "Change HSV limit")

        #create mask
        lower_color = np.array([l_h,l_s,l_v])
        upper_color = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv,lower_color,upper_color)

        #find countors

        #change to mask!!!!!!!!!!!!!!!!!!!! or screen (depend)
        contours_list, hierarchy = cv2.findContours(mask,
                                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = []
        count = 0
        for contour in contours_list:
            x, y, w, h = cv2.boundingRect(contour)
            p1 = (x, y)
            p2 = (x + w, y + h)
            #print(contour)
            contours.append((p1, p2))
            #try different methods this single algorightm might fail
            approx = cv2.approxPolyDP(contour, (float(p_t))/1000 * cv2.arcLength(contour, True), True) #0.037
            cv2.drawContours(screen_final, [approx], 0, (0, 0, 255), 2)
            cv2.drawContours(screen_final, [contour], 0, (0, 255, 0), 1)
            #print(len(approx))
            if cv2.contourArea(contour) > cnt_area:
                if len(approx) == 3:
                    cv2.putText(screen_final, "Fuc", (approx.ravel()[0], approx.ravel()[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (0, 0, 255))
                elif len(approx) == 4:
                    cv2.putText(screen_final, "rec", (approx.ravel()[0], approx.ravel()[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (0, 0, 255))
                elif len(approx) > 4:
                    cv2.putText(screen_final, "cir", (approx.ravel()[0], approx.ravel()[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (0, 0, 255))


        cv2.imshow("screen", np.array(screen_final))

        cv2.imshow("mask", np.array(mask))





        last_time = time.time()
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    return

path = sys.argv[1]
#countmono(path)
print(path)
capturescreen()