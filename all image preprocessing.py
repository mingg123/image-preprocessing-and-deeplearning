import mritopng
import numpy as np
import cv2
import os
from PIL import Image
from matplotlib import pyplot as plt
import csv
import shutil


def dicom(path, save_path):
    print("dicom start!")
    #mritopng.convert_folder(r'C:\Users\user\PycharmProjects\dicom', path)
    png_list = []
    dcm_list = []

    for file in [doc for doc in os.listdir(path) if doc.endswith(".png")]:
        png_list.append(file)

    for file2 in [doc for doc in os.listdir(path) if doc.endswith(".dcm")]:
        dcm_list.append(file2)

    new_dir = path + '\\' + 'new'
    if not(os.path.isdir(new_dir)):
        os.makedirs(os.path.join(new_dir))


    for k in range(len(dcm_list)):
        dicom_path = path + '\\' + dcm_list[k]
        shutil.move(dicom_path, new_dir)

    mritopng.convert_folder(new_dir, save_path)


    for i in range(len(png_list)):
        png_path = path + '\\' + png_list[i]
        shutil.move(png_path, save_path)
        print(png_path + ' success')
#png1폴더는 생성하면안됨 만들지않고 지정해주어야함

def crop(img,name):
    print("crop start!")
    row, col = img.shape[:2]
    # print(row,col)
    row2 = row * 0.04
    col2 = col * 0.04
    #print(row2)
    dst = img.copy()
    dst1 = dst[int(row2):int(row - row2), int(col2):int(col - col2)]  # [세로, 가로]

    os.chdir(save_path)
    save_name = name[:len(name) - 4] + '_crop.png'
    cv2.imwrite(save_name, dst1)


def after_label(label_removed, img, save_path, name):
    img = cv2.resize(img, None, fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_AREA)
    row, col = img.shape[:2]

    result = img.copy()

    for i in range(0, row):
        for j in range(0, col):
            if (label_removed[i, j] == 0):
                result[i, j] = 0

    os.chdir(save_path)
    save_name = name[:len(name) - 8] + 'label.png'
    cv2.imwrite(save_name, result)


def label(img, name):
    # 0. Down sizing & gray scailing
    gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray0, None, fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_AREA)

    # 1. 이진화
    ret, result = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # 2. 픽셀이 min_size 이하인 것 제거

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(result, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    min_size = 100000

    img2 = np.zeros((output.shape))

    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    # 라벨이 제거된 이진화 값 저장
    # os.chdir(r'D:\test4')
    # save_name = name[:len(name)-8] + 'thres.png'
    # cv2.imwrite(save_name, img2)

    return img2

def muselright(img,name,justname):
    print("musel right start!")
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
    os.chdir(save_path1)
    # cv2.imwrite(name[:len(name) - 4] + '_musel4' + '.png', data3)

    thresh = Image.fromarray(data3)
    width, height = ori.size

    for i in range(width):
        for j in range(height):
            if (thresh.getpixel((i, j)) == (0, 0, 0)):
                ori.putpixel((i, j), (0, 0, 0))

    save_name = justname[:len(justname)]
    saveimg = np.array(ori)

    cv2.imwrite(save_name, saveimg)


def muselleft(img,name, justname):
    print("muselleft start!")
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
    os.chdir(save_path1)
    #cv2.imwrite(name[:len(name) - 4] + '_musel4' + '.png', data3)

    thresh = Image.fromarray(data3)
    width, height = ori.size


    for i in range(width):
        for j in range(height):
            if(thresh.getpixel((i,j)) == (0, 0, 0)):
                ori.putpixel((i,j), (0,0,0))


    save_name = justname[:len(justname)]
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
    if(mean1<mean2): #왼쪽 더 크면 밝다는 이야기는 그쪽에 근육이있음
        print('Left')  #왼쪽에 근육이있다.
        muselleft(img,name, justname)
    else :
        print('Right')
        muselright(img,name,justname)


def alarm(data, name):
    print("alarm start!")
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
    os.chdir(save_path2)
    cv2.imwrite(str(name[:len(name) - 9]) +'ALA.png', im_out)



def morphology(img, save_path, name):
    print("팽창 시작!")
    col, row = img.shape[:2]
    kernel1 = np.ones((10, 10), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel1)
    # 오프닝

    kernel2 = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(opening, kernel2, iterations=1)

    save = save_path + '\\' + name
    cv2.imwrite(save, img_dilation)


def make(data1, data2, save_path,save_path1):
    print("합성시작!")
    seg_read = cv2.imread(save_path1 + "\\" + data1, cv2.COLOR_BGR2GRAY)
    img_read = cv2.imread(save_path1 + "\\" + data2, cv2.COLOR_BGR2GRAY)
    seg_read1 = cv2.cvtColor(seg_read, cv2.COLOR_BGR2GRAY)
    img_read1 = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)

    row, col = seg_read.shape[:2]
    result = img_read1.copy()

    for i in range(0, row):
        for j in range(0, col):
            if((seg_read1[i,j]) == 0):
                result[i,j] = 0

    os.chdir(save_path)
    cv2.imwrite(data1[:len(data1)-7] + 'SEG.png', result)


def ehd(img, path, name):
    row, col = img.shape[:2]
    sigma = [3, 6, 12, 24, 30]
    f2 = [4, 16, 64, 128, 200]
    x = np.zeros((4, 3, 3))
    x[0] = [[-1, 0, 1], [-2, 0, 2], [-1, 1, 1]]  # 0도
    x[1] = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]  # 45도
    x[2] = [[2, 2, -1], [2, -1, -1], [-1, -1, -1]]  # 90도
    x[3] = [[-1, -1, -1], [2, -1, -1], [2, 2, -1]]  # 135도

    tmax = [0, 0, 0, 0]
    tmaxmax = np.zeros(20)
    z = 0
    for y in range(0, 5):
        for k in range(0, 4):
            gk = cv2.getGaborKernel((int(row / 4), int(col / 4)), sigma[y], np.pi / 4 * 1, 2 * np.pi / f2[k], 1, 0,
                                    ktype=cv2.CV_32F)
            temp = cv2.filter2D(img, cv2.CV_8UC3, gk)
            for r in range(0, 4):
                img1 = cv2.filter2D(temp, cv2.CV_8UC3, x[r])
                data = np.array(img1)
                sortdata = np.sort(data, axis=None)
                sortdata1 = np.sort(sortdata)[::-1]  # 오름차순정렬
                max = sortdata1[0]  # 제일큰값임으로
                count = 0
                for i in range(0, len(sortdata1)):
                    if (max == sortdata1[i]):
                        count = count + 1  # 큰값과 같으면 count+1
                tmax[r] = count  # 최대강도픽셀수 계산

            maxValue = tmax[0]
            for i in range(1, len(tmax)):
                if maxValue < tmax[i]:
                    maxValue = tmax[i]
            tmaxmax[z] = maxValue
            z = z + 1
            print(tmaxmax)

    str1 = str(name[:len(name) - 7]) + '.png'
    mydata = [tmaxmax[0], tmaxmax[1], tmaxmax[2], tmaxmax[3], tmaxmax[4], tmaxmax[5], tmaxmax[6], tmaxmax[7],
              tmaxmax[8]
        , tmaxmax[9], tmaxmax[10], tmaxmax[11], tmaxmax[12], tmaxmax[13], tmaxmax[14], tmaxmax[15], tmaxmax[16],
              tmaxmax[17]
        , tmaxmax[18], tmaxmax[19], str1]

    csvfile = open(r'C:\Users\user\PycharmProjects\untitled5\data50.csv', "a", newline='')
    csvwrite = csv.writer(csvfile)
    csvwrite.writerow(mydata)
    csvfile.close()

def main(path0,path,save_path,save_path1,save_path2,save_path3):
    dicom(path0,path)

    file_list = []
    for file in [doc for doc in os.listdir(path) if doc.endswith(".png")]:
        file_list.append(file)

    for i in range(len(file_list)):
        print(file_list[i])
        name = path + '\\' + file_list[i]
        img = cv2.imread(name)
        result = crop(img, file_list[i])

    file_list = []
    for file in [doc for doc in os.listdir(save_path) if doc.endswith("crop.png")]:
        file_list.append(file)

    for k in range(len(file_list)):
        full_path = save_path + '\\' + file_list[k]
        # print(namee)
        img = cv2.imread(full_path)
        data = np.array(img)
        label_removed = label(data, file_list[k])
        after_label(label_removed, img, save_path1, file_list[k])
        print(file_list[k])

    file_list = []
    for file in [doc for doc in os.listdir(save_path1) if doc.endswith("MLO.LJPEG_label.png")]:
        file_list.append(file)

    for i in range(len(file_list)):
        print(file_list[i])
        name = save_path1 + '\\' + file_list[i]
        img = cv2.imread(name)
        remove(img, name, file_list[i])


    file_list = []
    for file in [doc for doc in os.listdir(save_path1) if doc.endswith(".png")]:
        file_list.append(file)

    for k in range(len(file_list)):
        name = save_path1 + '\\' + file_list[k]
        #print(namee)
        img = cv2.imread(name)
        data = np.array(img)
        alarm(data, file_list[k])


    file_list = []
    for file in [doc for doc in os.listdir(save_path2) if doc.endswith("ALA.png")]:
        file_list.append(file)

    for k in range(len(file_list)):
        full_path = save_path2 + '\\' + file_list[k]
        # print(namee)
        img = cv2.imread(full_path)
        morphology(img, save_path1, file_list[k])


    segment_list = []  # segment list 배열
    for file in [doc for doc in os.listdir(save_path1) if doc.endswith("ALA.png")]:
        segment_list.append(file)

    image_list = []  # GCN list 배열
    isthere_list = []  # segment를 가지고 있는 GCN의 배열, segment_list의 길이와 동일해야함 반드시
    for file in [doc for doc in os.listdir(save_path1) if doc.endswith("label.png")]:
        image_list.append(file)


    # isthere_list append 작업
    for i in range(0, len(segment_list)):
        seg = segment_list[i]
        seg_name = seg[:len(seg) - 7]

        for k in range(0, len(image_list)):
            img = image_list[k]
            img_name = img[:len(img) - 9]
            if (seg_name == img_name):
                isthere_list.append(image_list[k])


    for j in range(0, len(segment_list)):

        make(segment_list[j], isthere_list[j], save_path3, save_path1)
        print(str(j) + "번째 완료!")

        ########ehd 백터값 추출 시작

    file_list = []
    for file in [doc for doc in os.listdir(save_path3) if doc.endswith(".png")]:
        file_list.append(file)
    for k in range(len(file_list)):
        full_path = save_path3 + '\\' + file_list[k]
        # print(namee)
        img = cv2.imread(full_path)
        ehd(img, save_path3, file_list[k])

path0=r'C:\Users\user\PycharmProjects\dicom'
path = r'C:\Users\user\PycharmProjects\s1'
save_path = r'C:\Users\user\PycharmProjects\crop'
save_path1= r'C:\Users\user\PycharmProjects\label'
save_path2=r'C:\Users\user\PycharmProjects\alam'
save_path3=r'C:\Users\user\PycharmProjects\sample1'

main(path0,path,save_path,save_path1,save_path2,save_path3)