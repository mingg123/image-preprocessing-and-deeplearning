import cv2

img = cv2.imread('LMLO.dcm_label.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('LMLO.dcm_label.png',img)  #grascal 시킨 이미지를 씀