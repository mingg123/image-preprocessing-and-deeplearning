import cv2
import numpy as np
import os

def alarm(data, name):

    print(name)
    # 평균구하기
    mean = cv2.mean(data)

    # 오름차순 정렬
    sortdata = np.sort(data, axis=None)
    sortdata1 = np.sort(sortdata)[::-1]


    if (mean < (40, 40, 40)):
        threshold = sortdata1[int(0.013 * len(sortdata))]

    elif (mean > (40, 40, 40) and mean < (60, 60, 60)):
        threshold = sortdata1[int(0.012 * len(sortdata))]

    else:
        threshold = sortdata1[int(0.009 * len(sortdata))]

    new_value = 255
    black_value = 0

    mask = data < threshold
    data[mask] = black_value


    data1 = np.array(data)
    imgray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, threshold, 255, 0)
    (contours, _) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(data1, contours, -1, (0, 0, 0), 1)

    # filling
    im_in = cv2.cvtColor(data1, cv2.COLOR_BGR2GRAY);

    th, im_th = cv2.threshold(im_in, threshold, 255, cv2.THRESH_TOZERO);

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    # alarm segment 저장 경로
    os.chdir(r"E:\cancer\figment.csee.usf.edu\pub\DDSM\cases\cancers\sample3")
    cv2.imwrite(str(name[:len(name) - 7]) +'ALA.png', im_out)

def Decom_all(path):
    file_list = []
    for file in [doc for doc in os.listdir(path) if doc.endswith(".png")]:
        file_list.append(file)

    for k in range(len(file_list)):
        name = path + '\\' + file_list[k]
        #print(namee)
        img = cv2.imread(name)
        data = np.array(img)
        alarm(data, file_list[k])


path = r'E:\cancer\figment.csee.usf.edu\pub\DDSM\cases\cancers\sample5'
Decom_all(path)
