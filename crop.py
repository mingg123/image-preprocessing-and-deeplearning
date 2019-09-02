import cv2
import os
import numpy as np

def make_crop_image(x, y, img, file_list, t):
    save_path = r'E:\cancer\figment.csee.usf.edu\pub\DDSM\cases\cancers\crop1'

    h = x
    w = y +224
    h2 = x + 224
    w2 = y

    #h = x - 224
    #w = y - 224
    #h2 = x
    #w2 = y
    crop_img = img[h:h2, w:w2]

    fname = file_list[:len(file_list) - 10] + '_crop' + str(t) + '.png'

    save_name = save_path + '\\' + fname
    cv2.imwrite(save_name, crop_img)
    print(save_name + '....')

def overla(f, img, file_list):
    i = 0
    line = f.readline()
    line = line.split(' ')
    n = int(line[1])
    #print(n)
    row, col = img.shape[:2]


    result = []
    t = 0
    while t < n:

        while 1:
            line = f.readline()
            line = line.split(' ')
            if line[0] == "TOTAL_OUTLINES":
                break

        line2 = f.readline()
        num = int(line[1])
        #print(num)
        line2 = f.readline()
        line2 = line2.split(' ')
        count = 2
        x = int(line2[1])
        y = int(line2[0])
        make_crop_image(x, y, img, file_list, t)

        #print(x, y)
        #print(line2)
        tempimg = np.zeros((row, col))
        while 1:
            tempimg[x, y] = 250
            temp = line2[count]

            if temp == "#\n":
                break
            if temp == "0":
                x = x - 1
            if temp == "1":
                y = y + 1
                x = x - 1
            if temp == "2":
                y = y + 1
            if temp == "3":
                x = x + 1
                y = y + 1
            if temp == "4":
                x = x + 1
            if temp == "5":
                x = x + 1
                y = y - 1
            if temp == "6":
                y = y - 1
            if temp == "7":
                x = x - 1
                y = y - 1

            count = count + 1
        k = 1
        while k < num:
            line2 = f.readline()
            line2 = f.readline()
            k = k + 1

        result.append(tempimg)
        t = t + 1



    k = 0
    while k < n:
        i = 0
        img5 = result[k]
        while i < row:
            j = 0
            a = False
            b = 0
            while j < col:
                if img5[i, j] == 250:
                    if b == j - 1:
                        a = a
                        b = j
                    else:
                        a = not a
                        b = j

                if a:
                    img5[i, j] = 250
                j = j + 1
            i = i + 1

        result[k] = img5
        strtemp = str(file_list)[:len(file_list) - 4] + "_over" + str(k) + ".jpg"
        cv2.imwrite(path + "\\" + strtemp, result[k])
        k = k + 1


    return result, k



def main(path):
    file_list = []      # .png 파일 리스트
    for file in [doc for doc in os.listdir(path) if doc.endswith(".png")]:
        file_list.append(file)

    for k in range(len(file_list)):
        name = path + '\\' + file_list[k]      # 경로 + 사진의 이름
        #print(name)
        img = cv2.imread(name)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if os.path.isfile(name[:len(name) - 9] + 'OVERLAY.txt'):
            name2 = name[:len(name)-9] + 'OVERLAY.txt'
            #print(name2)
            f = open(name2, 'r')

            print('overlay start')
            result, n = overla(f, img, file_list[k])
            print('overlay end')

for i in range(17, 99):
    path = r'E:\cancer\figment.csee.usf.edu\pub\DDSM\cases\cancers\cancer_02\case00' + str(i)
    main(path)