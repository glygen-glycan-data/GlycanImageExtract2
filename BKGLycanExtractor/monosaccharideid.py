import cv2, math, logging
import numpy as np

logger = logging.getLogger("search")

class MonoID:
    def __init__(self, **kw):
        pass

    def compare2img(self,img1,img2):
        # return similarity between two image
        if img1.shape == img2.shape:
            # print("samesize")
            # print(img1.shape)
            pass
        else:
            # print("img1,img2,not same size")
            # print(img1.shape)
            # print(img2.shape)
            return -1
        score = 0
        diff = cv2.absdiff(img1, img2)
        r, g, b = cv2.split(diff)
        score = cv2.countNonZero(g) / (img1.shape[0] * img1.shape[1])

        # cv2.imshow("different", diff)
        # cv2.waitKey(0)
        return 1 - score
        
    def compstr(self,counts):
        s = ""
        for sym,count in sorted(counts.items()):
            if count > 0:
                s += "%s(%d)"%(sym,count)
        return s
    def id_monos(self,**kw):
        raise NotImplementedError
        
class HeuristicMonos(MonoID):
    def __init__(self,colors = None):
        super().__init__()
        self.color_range = colors
    def crop_largest(self,image = []):
        img = image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
        contours_list, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #print(contours_list)
        area_list = []
        for i, contour in enumerate(contours_list):
            area = cv2.contourArea(contour)
            area_list.append((area, i))
        (_, largest_index) = max(area_list)
        out = np.zeros_like(img)
        cv2.drawContours(out, contours_list, largest_index, (255, 255, 255), -1)
        _, out = cv2.threshold(out, 230, 255, cv2.THRESH_BINARY_INV)

        out2 = cv2.bitwise_or(out, img)
        return out2
    def id_monos(self,image = None):
        monos = {}
        img = self.crop_largest(image)

        # print(img_file.shape[0]*img_file.shape[1])
        bigwhite = np.zeros([img.shape[0] + 30, img.shape[1] + 30, 3], dtype=np.uint8)
        bigwhite.fill(255)
        bigwhite[15:15 + img.shape[0], 15:15 + img.shape[1]] = img
        img = bigwhite.copy()



        mag = 84000 / (img.shape[0] * img.shape[1])
        # print(mag)
        if mag <= 1:
            mag = 1
        img = cv2.resize(img, None, fx=mag, fy=mag)
        img = cv2.GaussianBlur(img, (11, 11), 0)
        # _,img_file=cv2.threshold(img_file,140,255,cv2.THRESH_BINARY)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_width = img.shape[0]
        img_height = img.shape[1]

        # read color range in config folder
        #origin = img.copy()
        final = img.copy()  # final annotated pieces

        #d = {}
        color_range_dict = self.color_range
        # color_range_file = open(colors_range)
        # color_range_dict = {}
        # for line in color_range_file.readlines():
        #     line = line.strip()
        #     name = line.split("=")[0].strip()
        #     color_range = line.split("=")[1].strip()
        #     color_range_dict[name] = np.array(list(map(int, color_range.split(","))))

        # create mask for each color
        yellow_mask = cv2.inRange(hsv, color_range_dict['yellow_lower'], color_range_dict['yellow_upper'])
        purple_mask = cv2.inRange(hsv, color_range_dict['purple_lower'], color_range_dict['purple_upper'])
        red_mask_l = cv2.inRange(hsv, color_range_dict['red_lower_l'], color_range_dict['red_upper_l'])
        red_mask_h = cv2.inRange(hsv, color_range_dict['red_lower_h'], color_range_dict['red_upper_h'])
        red_mask = red_mask_l + red_mask_h
        green_mask = cv2.inRange(hsv, color_range_dict['green_lower'], color_range_dict['green_upper'])
        blue_mask = cv2.inRange(hsv, color_range_dict['blue_lower'], color_range_dict['blue_upper'])
        black_mask = cv2.inRange(hsv, color_range_dict['black_lower'], color_range_dict['black_upper'])

        # store these mask into array
        mask_array = (red_mask, yellow_mask, green_mask, blue_mask, purple_mask, black_mask)
        mask_array_name = ("red_mask", "yellow_mask", "green_mask", "blue_mask", "purple_mask", "black_mask")
        mask_dict = dict(zip(mask_array_name, mask_array))
        # all_mask = sum(mask_array)

        # loop through each countors
        monoCount_dict = {"GlcNAc": 0, "NeuAc": 0, "Fuc": 0, "Man": 0, "GalNAc": 0, "Gal": 0,"Glc": 0, }
        yellows_contours = []
        return_contours = []
        for color in mask_array_name:
            if color == "black_mask":
                continue
            contours_list, _ = cv2.findContours(mask_dict[color],
                                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = []
            for contour in contours_list:

                approx = cv2.approxPolyDP(contour, 0.035 * cv2.arcLength(contour, True), True)
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                squareness = abs(math.log(float(w)/float(h),2))
                arearatio = 1e6*float(area)/(img_height*img_width)
                arearatio1 = 1000*area/float(w*h)
                if squareness < 2 and arearatio > 100 and arearatio1 > 200:
                    logger.info(f"{x,y,w,h,area,round(squareness,2),round(arearatio,2),round(arearatio1,2),color} ")
                    if squareness > 0.25 or arearatio < 1000.0 or arearatio1 < 500:
                        logger.info("BAD")
                        continue
                    p1 = (x, y)
                    p2 = (x + w, y + h)
                    contours.append((p1, p2))
                    cv2.rectangle(final, p1, p2, (0, 255, 0), 1)
                    cv2.drawContours(final, [approx], 0, (0, 0, 255), 1)

                    if color == "red_mask":
                        cv2.putText(final, "Fuc", (approx.ravel()[0], approx.ravel()[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    (0, 0, 255))
                        monoCount_dict["Fuc"] += 1
                        return_contours.append(("Fuc", contour))
                        logger.info("Fuc")

                    elif color == "purple_mask":
                        cv2.putText(final, "NeuAc", (approx.ravel()[0], approx.ravel()[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1,
                                    (0, 0, 255))
                        monoCount_dict["NeuAc"] += 1
                        return_contours.append(("NeuAc", contour))
                        logger.info("NeuAc")

                    elif color == "blue_mask":
                        white = np.zeros([h, w, 3], dtype=np.uint8)
                        white.fill(255)
                        this_blue_img = blue_mask[y:y + h, x:x + w]
                        this_blue_img = cv2.cvtColor(this_blue_img, cv2.COLOR_GRAY2BGR)
                        score = self.compare2img(white, this_blue_img)
                        if score >= 0.8:  # is square
                            cv2.putText(final, "GlcNAc", (approx.ravel()[0], approx.ravel()[1]),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                        (0, 0, 255))
                            monoCount_dict["GlcNAc"] += 1
                            return_contours.append(("GlcNAc", contour))
                            logger.info("GlcNAc")

                        elif 0.5 < score < 0.8:
                            cv2.putText(final, "Glc", (approx.ravel()[0], approx.ravel()[1]),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                        (0, 0, 255))
                            monoCount_dict["Glc"] += 1
                            return_contours.append(("Glc", contour))
                            logger.info("Glc")

                        else:
                            cv2.putText(final, "?", (approx.ravel()[0], approx.ravel()[1]),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                        (0, 0, 255))
                            logger.info("???")
                        #cv2.imshow("blue_mask",this_blue_img)
                        #cv2.imshow("origin_mask", origin[y:y + h, x:x + w])
                        #cv2.waitKey(0)
                    elif color == "green_mask":
                        cv2.putText(final, "Man", (approx.ravel()[0], approx.ravel()[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    (0, 0, 255))
                        monoCount_dict["Man"] += 1
                        return_contours.append(("Man", contour))
                        logger.info("Man")

                    elif color == "yellow_mask":

                        yellows_contours.append(contour)
                        white = np.zeros([h, w, 3], dtype=np.uint8)
                        white.fill(255)
                        this_yellow_img = yellow_mask[y:y + h, x:x + w]
                        # this_yellow_img = cv2.resize(this_yellow_img, None, fx=1, fy=1)
                        this_yellow_img = cv2.cvtColor(this_yellow_img, cv2.COLOR_GRAY2BGR)

                        score = self.compare2img(white, this_yellow_img)
                        if score > 0.9:  # is square
                            cv2.putText(final, "GalNAc", (approx.ravel()[0], approx.ravel()[1]),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                        (0, 0, 255))
                            monoCount_dict["GalNAc"] += 1
                            return_contours.append(("GalNAc", contour))
                            logger.info("GalNAc")

                        elif 0.5 < score < 0.9:
                            cv2.putText(final, "Gal", (approx.ravel()[0], approx.ravel()[1]),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                        (0, 0, 255))
                            monoCount_dict["Gal"] += 1
                            return_contours.append(("Gal", contour))
                            logger.info("Gal")

                        else:
                            cv2.putText(final, "?", (approx.ravel()[0], approx.ravel()[1]),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                        (0, 0, 255))
                            logger.info("???",score)

        #     pass
        # print("herte",yellows_contours)
        # cv2.imshow("yellow_mask",all_mask)
        # cv2.imshow("final", final)
        # cv2.waitKey(0)
        monos["count_dict"] = monoCount_dict
        monos["annotated"] = final
        monos["mask_dict"] = mask_dict
        monos["contours"] = return_contours
        return monos
    