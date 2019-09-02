import numpy as np
import cv2
import os
import csv

def main(img, path, name):
    #img = cv2.imread('C_0001_1.RIGHT_CC.LJPEG_GCN1_ALA1.png')
    row, col = img.shape[:2]
    print(row)
    print(col)
    sigma = [3, 6, 12, 24, 30]
    f = [6, 12, 24, 48, 80]
    f2 = [1, 2, 4, 8, 16]
    x = np.zeros((4, 3, 3))
    x[0] = [[-1, 0, 1], [-2, 0, 2], [-1, 1, 1]]  # 0도
    x[1] = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]  # 45도
    x[2] = [[2, 2, -1], [2, -1, -1], [-1, -1, -1]]  # 90도
    x[3] = [[-1, -1, -1], [2, -1, -1], [2, 2, -1]]  # 135도

    tmax = [0, 0, 0, 0]
    tmaxmax = np.zeros(20)
    tmaxlength = len(tmaxmax)
    z = 0
    for y in range(0, 5):
        gk = cv2.getGaborKernel((int(row / 4), int(col / 4)), sigma[y], np.pi / 4 * 1, 2 * np.pi / f2[0], 1, 0,
                                ktype=cv2.CV_32F)
        temp = cv2.filter2D(img, cv2.CV_8UC3, gk)
        cv2.imwrite("test0.png", temp)

        img1 = cv2.filter2D(temp, cv2.CV_8UC3, x[0])
        img2 = cv2.filter2D(temp, cv2.CV_8UC3, x[1])
        img3 = cv2.filter2D(temp, cv2.CV_8UC3, x[2])
        img4 = cv2.filter2D(temp, cv2.CV_8UC3, x[3])

        data = np.array(img1)
        sortdata = np.sort(data, axis=None)
        sortdata1 = np.sort(sortdata)[::-1]
        # print(sortdata1)
        max = sortdata1[0]
        # print(len(sortdata1))
        count = 0
        for i in range(0, len(sortdata1)):
            if (max == sortdata1[i]):
                count = count + 1
        #print(count)
        tmax[0] = count

        data = np.array(img2)
        sortdata = np.sort(data, axis=None)
        sortdata1 = np.sort(sortdata)[::-1]
        # print(sortdata1)
        max = sortdata1[0]
        # print(len(sortdata1))
        count = 0
        for i in range(0, len(sortdata1)):
            if (max == sortdata1[i]):
                count = count + 1
        #print(count)
        tmax[1] = count

        data = np.array(img3)
        sortdata = np.sort(data, axis=None)
        sortdata1 = np.sort(sortdata)[::-1]
        # print(sortdata1)
        max = sortdata1[0]
        # print(len(sortdata1))
        count = 0
        for i in range(0, len(sortdata1)):
            if (max == sortdata1[i]):
                count = count + 1
        #print(count)
        tmax[2] = count

        data = np.array(img4)
        sortdata = np.sort(data, axis=None)
        sortdata1 = np.sort(sortdata)[::-1]
        # print(sortdata1)
        max = sortdata1[0]
        # print(len(sortdata1))
        count = 0
        for i in range(0, len(sortdata1)):
            if (max == sortdata1[i]):
                count = count + 1
        #print(count)
        tmax[3] = count

        #print(tmax)

        maxValue = tmax[0]
        for i in range(1, len(tmax)):
            if maxValue < tmax[i]:
                maxValue = tmax[i]
        #print(maxValue)
        tmaxmax[z] = maxValue
        z = z + 1

        # print(tmaxmax)
        #######################################################################################################################
        gk1 = cv2.getGaborKernel((int(row / 4), int(col / 4)), sigma[y], np.pi / 4 * 1, 2 * np.pi / f2[1], 1, 0,
                                 ktype=cv2.CV_32F)
        temp1 = cv2.filter2D(img, cv2.CV_8UC3, gk1)
        cv2.imwrite("test0.png", temp1)

        img1 = cv2.filter2D(temp1, cv2.CV_8UC3, x[0])
        img2 = cv2.filter2D(temp1, cv2.CV_8UC3, x[1])
        img3 = cv2.filter2D(temp1, cv2.CV_8UC3, x[2])
        img4 = cv2.filter2D(temp1, cv2.CV_8UC3, x[3])

        data = np.array(img1)
        sortdata = np.sort(data, axis=None)
        sortdata1 = np.sort(sortdata)[::-1]
        # print(sortdata1)
        max = sortdata1[0]
        # print(len(sortdata1))
        count = 0
        for i in range(0, len(sortdata1)):
            if (max == sortdata1[i]):
                count = count + 1
        #print(count)
        tmax[0] = count

        data = np.array(img2)
        sortdata = np.sort(data, axis=None)
        sortdata1 = np.sort(sortdata)[::-1]
        # print(sortdata1)
        max = sortdata1[0]
        # print(len(sortdata1))
        count = 0
        for i in range(0, len(sortdata1)):
            if (max == sortdata1[i]):
                count = count + 1
        #print(count)
        tmax[1] = count

        data = np.array(img3)
        sortdata = np.sort(data, axis=None)
        sortdata1 = np.sort(sortdata)[::-1]
        # print(sortdata1)
        max = sortdata1[0]
        # print(len(sortdata1))
        count = 0
        for i in range(0, len(sortdata1)):
            if (max == sortdata1[i]):
                count = count + 1
        #print(count)
        tmax[2] = count

        data = np.array(img4)
        sortdata = np.sort(data, axis=None)
        sortdata1 = np.sort(sortdata)[::-1]
        # print(sortdata1)
        max = sortdata1[0]
        # print(len(sortdata1))
        count = 0
        for i in range(0, len(sortdata1)):
            if (max == sortdata1[i]):
                count = count + 1
        #print(count)
        tmax[3] = count

        #print(tmax)

        maxValue = tmax[0]
        for i in range(1, len(tmax)):
            if maxValue < tmax[i]:
                maxValue = tmax[i]
        #print(maxValue)
        tmaxmax[z] = maxValue
        z = z + 1

        gk2 = cv2.getGaborKernel((int(row / 4), int(col / 4)), sigma[y], np.pi / 4 * 1, 2 * np.pi / f2[2], 1, 0,
                                 ktype=cv2.CV_32F)
        temp = cv2.filter2D(img, cv2.CV_8UC3, gk2)
        cv2.imwrite("test0.png", temp)

        img1 = cv2.filter2D(temp, cv2.CV_8UC3, x[0])
        img2 = cv2.filter2D(temp, cv2.CV_8UC3, x[1])
        img3 = cv2.filter2D(temp, cv2.CV_8UC3, x[2])
        img4 = cv2.filter2D(temp, cv2.CV_8UC3, x[3])

        data = np.array(img1)
        sortdata = np.sort(data, axis=None)
        sortdata1 = np.sort(sortdata)[::-1]
        # print(sortdata1)
        max = sortdata1[0]
        # print(len(sortdata1))
        count = 0
        for i in range(0, len(sortdata1)):
            if (max == sortdata1[i]):
                count = count + 1
        #print(count)
        tmax[0] = count

        data = np.array(img2)
        sortdata = np.sort(data, axis=None)
        sortdata1 = np.sort(sortdata)[::-1]
        # print(sortdata1)
        max = sortdata1[0]
        # print(len(sortdata1))
        count = 0
        for i in range(0, len(sortdata1)):
            if (max == sortdata1[i]):
                count = count + 1
        #print(count)
        tmax[1] = count

        data = np.array(img3)
        sortdata = np.sort(data, axis=None)
        sortdata1 = np.sort(sortdata)[::-1]
        # print(sortdata1)
        max = sortdata1[0]
        # print(len(sortdata1))
        count = 0
        for i in range(0, len(sortdata1)):
            if (max == sortdata1[i]):
                count = count + 1
        #print(count)
        tmax[2] = count

        data = np.array(img4)
        sortdata = np.sort(data, axis=None)
        sortdata1 = np.sort(sortdata)[::-1]
        # print(sortdata1)
        max = sortdata1[0]
        # print(len(sortdata1))
        count = 0
        for i in range(0, len(sortdata1)):
            if (max == sortdata1[i]):
                count = count + 1
        #print(count)
        tmax[3] = count

        #print(tmax)

        maxValue = tmax[0]
        for i in range(1, len(tmax)):
            if maxValue < tmax[i]:
                maxValue = tmax[i]
        #print(maxValue)
        tmaxmax[z] = maxValue
        z = z + 1

        # print(tmaxmax)
        gk3 = cv2.getGaborKernel((int(row / 4), int(col / 4)), sigma[y], np.pi / 4 * 1, 2 * np.pi / f2[3], 1, 0,
                                 ktype=cv2.CV_32F)
        temp3 = cv2.filter2D(img, cv2.CV_8UC3, gk3)
        cv2.imwrite("test0.png", temp3)

        img1 = cv2.filter2D(temp3, cv2.CV_8UC3, x[0])
        img2 = cv2.filter2D(temp3, cv2.CV_8UC3, x[1])
        img3 = cv2.filter2D(temp3, cv2.CV_8UC3, x[2])
        img4 = cv2.filter2D(temp3, cv2.CV_8UC3, x[3])

        data = np.array(img1)
        sortdata = np.sort(data, axis=None)
        sortdata1 = np.sort(sortdata)[::-1]
        # print(sortdata1)
        max = sortdata1[0]
        # print(len(sortdata1))
        count = 0
        for i in range(0, len(sortdata1)):
            if (max == sortdata1[i]):
                count = count + 1
        #print(count)
        tmax[0] = count

        data = np.array(img2)
        sortdata = np.sort(data, axis=None)
        sortdata1 = np.sort(sortdata)[::-1]
        # print(sortdata1)
        max = sortdata1[0]
        # print(len(sortdata1))
        count = 0
        for i in range(0, len(sortdata1)):
            if (max == sortdata1[i]):
                count = count + 1
        #print(count)
        tmax[1] = count

        data = np.array(img3)
        sortdata = np.sort(data, axis=None)
        sortdata1 = np.sort(sortdata)[::-1]
        # print(sortdata1)
        max = sortdata1[0]
        # print(len(sortdata1))
        count = 0
        for i in range(0, len(sortdata1)):
            if (max == sortdata1[i]):
                count = count + 1
        #print(count)
        tmax[2] = count

        data = np.array(img4)
        sortdata = np.sort(data, axis=None)
        sortdata1 = np.sort(sortdata)[::-1]
        # print(sortdata1)
        max = sortdata1[0]
        # print(len(sortdata1))
        count = 0
        for i in range(0, len(sortdata1)):
            if (max == sortdata1[i]):
                count = count + 1
        #print(count)
        tmax[3] = count

        #print(tmax)

        maxValue = tmax[0]
        for i in range(1, len(tmax)):
            if maxValue < tmax[i]:
                maxValue = tmax[i]
        #print(maxValue)
        tmaxmax[z] = maxValue
        z = z + 1

        print(tmaxmax)


    csvfile = open(r'C:\Users\user\PycharmProjects\untitled5\iris.data.csv', "a", newline='')
    csvwrite = csv.writer(csvfile)
    csvwrite.writerow(tmaxmax)
    csvfile.close()


def Decom_all(path):
        file_list = []
        for file in [doc for doc in os.listdir(path) if doc.endswith(".png")]:
            file_list.append(file)
        for k in range(len(file_list)):
            full_path = path + '\\' + file_list[k]
            # print(namee)
            img = cv2.imread(full_path)
            main(img, path, file_list[k])
for i in range(3, 4):
    path = r'E:\cancer\figment.csee.usf.edu\pub\DDSM\cases\cancers\sample' + str(i)
    Decom_all(path)
