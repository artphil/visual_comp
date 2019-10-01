import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

image = cv.imread('cena1.png')
gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)


# plt.imshow(gray, cmap='gray')

cv.imshow('dst',gray)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()

(T, binary) = cv.threshold(gray, 50, 255, cv.THRESH_BINARY_INV)
(H, compare) = cv.threshold(gray, 50, 255, cv.THRESH_BINARY_INV)

fig, (img1, img2) = plt.subplots(1, 2)

# img1.imshow(binary, cmap='gray')
# img2.imshow(compare, cmap='gray')

cv.imshow('dst',binary)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()

edges = cv.Canny(binary,100,200)

plt.imshow(edges, cmap='gray')

imggray = np.float32(binary)
imggrayCRN = np.float32(compare)

corners = cv.cornerHarris(binary,3,3,0.04)
compareCRN = cv.cornerHarris(imggrayCRN,3,25,0.04)

cv.imshow('dst',corners)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()



fig, (crn1, crn2) = plt.subplots(1, 2)

crn1.imshow(corners, cmap='gray')
crn2.imshow(compareCRN, cmap='gray')

print(corners.shape[0])
print(corners.shape[1])

image[corners>0.01*corners.max()]=[255,0,0]

plt.imshow(image, cmap='gray')

target = cv.imread('alvo.jpg')
t_gray = cv.cvtColor(target,cv.COLOR_BGR2GRAY)
(T, t_bin) = cv.threshold(t_gray, 100, 200, cv.THRESH_BINARY_INV)
t_gray = np.float32(t_bin)
t_corners = cv.cornerHarris(t_gray,2,3,0.04)
target[t_corners>0.01*t_corners.max()]=[255,0,0]

plt.imshow(target, cmap='gray')

for row in range(len(corners)):
    for col in range(len(corners[0])):
        if corners[row,col] > 4000000:
            print ("({},{}): {}".format(row,col,corners[row,col]))
    
    
