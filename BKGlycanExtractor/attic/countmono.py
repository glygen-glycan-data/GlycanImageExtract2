import cv2, sys,os
import ImageGrab # used in capturescreen function
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
            cv2.drawContours(screen_final, contour, 0, (0, 255, 0), 1)
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
        red1=cv2.inRange(hsv,np.array([0,18,120]),np.array([19,255,255]))
        red2=cv2.inRange(hsv,np.array([156,40,120]),np.array([180,255,255]))
        redmask=red1+red2

        cv2.imshow("redmask",np.array(redmask))
        cv2.imshow("mask", np.array(mask))







        last_time = time.time()
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    return

def compare2img(img1,img2):
    if img1.shape == img2.shape:
        #print("samesize")
        #print(img1.shape)
        pass
    else:
        print("img1,img2,not same size")
        print(img1.shape)
        print(img2.shape)
        return -1
    score =0
    diff = cv2.absdiff(img1,img2)
    r,g,b=cv2.split(diff)
    score=cv2.countNonZero(g)/(img1.shape[0]*img1.shape[1])

    #cv2.imshow("different", diff)
    #cv2.waitKey(0)
    return 1-score


def countcolors(img_file):
    #process image input
    mag=4
    img_file=cv2.resize(img_file,None,fx=mag,fy=mag)
    hsv=cv2.cvtColor(img_file,cv2.COLOR_BGR2HSV)


    #read color range in config folder
    final = img_file.copy()
    d = {}
    color_range_file = open("configs\colors_range.txt")
    color_range_dict={}
    for line in color_range_file.readlines():
        line=line.strip()
        name = line.split("=")[0].strip()
        color_range = line.split("=")[1].strip()
        color_range_dict[name]=np.array(list(map(int,color_range.split(","))))



    #create mask for each color
    yellow_mask=cv2.inRange(hsv,color_range_dict['yellow_lower'],color_range_dict['yellow_upper'])
    purple_mask=cv2.inRange(hsv,color_range_dict['purple_lower'],color_range_dict['purple_upper'])
    red_mask_l=cv2.inRange(hsv,color_range_dict['red_lower_l'],color_range_dict['red_upper_l'])
    red_mask_h = cv2.inRange(hsv, color_range_dict['red_lower_h'], color_range_dict['red_upper_h'])
    red_mask=red_mask_l+red_mask_h
    green_mask=cv2.inRange(hsv,color_range_dict['green_lower'],color_range_dict['green_upper'])
    blue_mask=cv2.inRange(hsv,color_range_dict['blue_lower'],color_range_dict['blue_upper'])
    black_mask = cv2.inRange(hsv,color_range_dict['black_lower'],color_range_dict['black_upper'])
    cv2.imwrite(f"test/black_mask.png", black_mask)
    #store these mask into array
    mask_array=(red_mask,yellow_mask,green_mask,blue_mask,purple_mask)
    mask_array_name=("red_mask","yellow_mask","green_mask","blue_mask","purple_mask")
    mask_dict = dict(zip(mask_array_name,mask_array))
    all_mask = sum(mask_array)

    #loop through each countors
    monoCount_dict={"GlcNAc":0,"NeuAc":0,"Fuc":0,"Man":0,"GalNAc":0,"Gal":0,}
    yellows_contours = []
    for color in mask_array_name:
        contours_list, _ = cv2.findContours(mask_dict[color],
                                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = []

        for contour in contours_list:

            approx = cv2.approxPolyDP(contour, 0.035 * cv2.arcLength(contour, True), True)
            if cv2.contourArea(contour)>60*mag:  # need find average size for mono
                x, y, w, h = cv2.boundingRect(contour)
                p1 = (x, y)
                p2 = (x + w, y + h)
                contours.append((p1, p2))
                cv2.rectangle(final, p1, p2, (0, 255, 0), 1)
                cv2.drawContours(final, [approx], 0, (0, 0, 255), 1)

                if color == "red_mask":
                    cv2.putText(final, "Fuc", (approx.ravel()[0], approx.ravel()[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (0, 0, 255))
                    monoCount_dict["Fuc"] +=1
                elif color == "purple_mask":
                    cv2.putText(final, "NeuAc", (approx.ravel()[0], approx.ravel()[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (0, 0, 255))
                    monoCount_dict["NeuAc"]+=1
                elif color == "blue_mask":
                    cv2.putText(final, "GlcNAc", (approx.ravel()[0], approx.ravel()[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (0, 0, 255))
                    monoCount_dict["GlcNAc"]+=1
                elif color == "green_mask":
                    cv2.putText(final, "Man", (approx.ravel()[0], approx.ravel()[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (0, 0, 255))
                    monoCount_dict["Man"]+=1
                elif color == "yellow_mask":

                    yellows_contours.append(contour)
                    white = np.zeros([h, w,3], dtype=np.uint8)
                    white.fill(255)
                    this_yellow_img = yellow_mask[y:y+h,x:x+w]
                    this_yellow_img=cv2.resize(this_yellow_img,None,fx=1,fy=1)
                    this_yellow_img=cv2.cvtColor(this_yellow_img,cv2.COLOR_GRAY2BGR)

                    score = compare2img(white,this_yellow_img)
                    if score >0.9: #is square
                        cv2.putText(final, "GalNAc", (approx.ravel()[0], approx.ravel()[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (0, 0, 255))
                        monoCount_dict["GalNAc"]+=1
                    elif 0.6<score<0.9:
                        cv2.putText(final, "Gal", (approx.ravel()[0], approx.ravel()[1]),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    (0, 0, 255))
                        monoCount_dict["Gal"]+=1
                    else:
                        cv2.putText(final, "?", (approx.ravel()[0], approx.ravel()[1]),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    (0, 0, 255))

        pass
    #print("herte",yellows_contours)
    #cv2.imshow("yellow_mask",all_mask)
    #cv2.imshow("final", final)
    #cv2.waitKey(0)
    return monoCount_dict,final



#path = sys.argv[1]
#img_file = cv2.imread(path)
#countmono(path)
#print(path)
#capturescreen()
#monocount_dict = countcolors(img_file)
#print(monocount_dict)