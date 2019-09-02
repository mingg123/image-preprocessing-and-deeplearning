from PIL import Image
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def muselright(img,name,justname):
    ori = Image.fromarray(img)
    print(name)

    threshold=1
    new_value = 255 #하얗게
    black_value = 0

    mask = img < threshold
    img[mask] = black_value
    mask = img >= threshold
    img[mask]= new_value

    data1 = np.array(img)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, threshold, 255, 0)
    (contours, _) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(data1, contours, -1, (0, 0, 0), 1)
    col,row = data1.shape[:2]
    data2=data1.copy()
    for xpoint in range(row,0,-1):
        mean= cv2.mean(data1[10][xpoint-10])  #x포인트 찾음
        print(mean , xpoint)
        if(mean<(30,30,30)):
            print(xpoint)
            break;

    for ypoint in range(10,col):
        mean1= cv2.mean(data1[ypoint][row-10])  #y포인트 찾음
        #print(mean , xpoint)
        if(mean1<(30,30,30)):  #흰색이 아닌부분 검은색인부분
            print(ypoint)
            break;

    cv2.line(data2,(row,ypoint),(xpoint,0),(255,0,0),30)
    #cv2.imwrite(name[:len(name) - 4] + '_musel3' + '.png', data2)

    data3 = data2.copy()

    for i in range(0,ypoint):
            cv2.line(data3, (row, ypoint-i), (xpoint+i, 0), (0, 0, 0), 30)

    #################################### 저장경로
    os.chdir(r'E:\cancer\sample4')
    # cv2.imwrite(name[:len(name) - 4] + '_musel4' + '.png', data3)

    thresh = Image.fromarray(data3)
    width, height = ori.size

    for i in range(width):
        for j in range(height):
            if (thresh.getpixel((i, j)) == (0, 0, 0)):
                ori.putpixel((i, j), (0, 0, 0))

    save_name = justname[:len(justname) - 7] + 'musel' + '.png'
    saveimg = np.array(ori)

    cv2.imwrite(save_name, saveimg)


def muselleft(img,name, justname):
    ori = Image.fromarray(img)
    print(name)

    threshold=1
    new_value = 255 #하얗게
    black_value = 0

    mask = img < threshold
    img[mask] = black_value
    mask = img >= threshold
    img[mask]= new_value

    data1 = np.array(img)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, threshold, 255, 0)
    (contours, _) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(data1, contours, -1, (0, 0, 0), 1)
    col,row = data1.shape[:2]
    data2=data1.copy()

    for xpoint in range(10,row):
        mean= cv2.mean(data1[10][xpoint])  #x포인트 찾음
        #print(mean , xpoint)
        if(mean<(30,30,30)):
            print(xpoint)
            break;

    for ypoint in range(10,col):
        mean1= cv2.mean(data1[ypoint][10])  #x포인트 찾음
        #print(mean , xpoint)
        if(mean1<(30,30,30)):  #흰색이 아닌부분 검은색인부분
            print(ypoint)
            break;

    cv2.line(data2,(0,ypoint),(xpoint,0),(255,0,0),30)
    #cv2.imwrite(name[:len(name) - 4] + '_musel3' + '.png', data2)

    data3 = data2.copy()

    for i in range(0,ypoint):
            cv2.line(data3, (0, ypoint-i), (xpoint-i, 0), (0, 0, 0), 30)

    #################################### 저장경로
    os.chdir(r'E:\cancer\sample4')
    #cv2.imwrite(name[:len(name) - 4] + '_musel4' + '.png', data3)

    thresh = Image.fromarray(data3)
    width, height = ori.size


    for i in range(width):
        for j in range(height):
            if(thresh.getpixel((i,j)) == (0, 0, 0)):
                ori.putpixel((i,j), (0,0,0))


    save_name = justname[:len(justname) - 7] + 'musel' + '.png'
    saveimg = np.array(ori)

    cv2.imwrite(save_name, saveimg)



def remove(img, name, justname):
    col, row = img.shape[:2]
    row2 = row * 0.1
    col2 = col * 0.1

    dst = img.copy()

    right= dst[int(row-row2):int(row),int(10):int(col2)]
    left=dst[int(0):int(row2),int(0):int(col2)]

    mean1= cv2.mean(right)
    mean2 = cv2.mean(left)
    print("left 평균",mean2)
    print("right평균",mean1)
    if(mean1<mean2): #왼쪽 더 크면 밝다는 이야기는 그쪽에 근육이있음
        print('Left')  #왼쪽에 근육이있다.
        muselleft(img,name, justname)
    else :
        print('Right')
        muselright(img,name,justname)

def main(path):
    file_list = []
    for file in [doc for doc in os.listdir(path) if doc.endswith("MLO.LJPEG_label.png")]:
        file_list.append(file)

    for i in range(len(file_list)):
        print(file_list[i])
        name = path + '\\' + file_list[i]
        img = cv2.imread(name)
        remove(img, name, file_list[i])

path = r'E:\cancer\sample2'
main(path)