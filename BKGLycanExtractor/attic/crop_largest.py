import cv2
import numpy as np




def croplargest(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray, 217, 255, cv2.THRESH_BINARY_INV)
    contours_list, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    area_list = []
    for i, contour in enumerate(contours_list):
        area = cv2.contourArea(contour)
        area_list.append((area, i))
    (_, largest_index) = max(area_list)
    out = np.zeros_like(img)
    cv2.drawContours(out, contours_list, largest_index, (255, 255, 255), -1)
    cv2.imshow("largest contour", out)
    cv2.waitKey(0)
    _, out = cv2.threshold(out, 230, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("threshold", out)
    cv2.waitKey(0)
    out2 = cv2.bitwise_or(out, img)
    return out2

if __name__ =='__main__':
    img = cv2.imread("save_origin.png")
    cv2.imshow("1", img)
    cv2.waitKey(0)
    img=croplargest(img)
    cv2.imshow("2",img)
    cv2.waitKey(0)