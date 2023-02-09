import cv2
import string
import random

def random_string(N):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))

def Two_Dim_Image_Split(img):
    h, w, channels = img.shape
    half = w // 2
    half2 = h // 2
    tl = img[:half2, :half]
    bl = img[half2:, :half]
    tr = img[:half2, half:]
    br = img[half2:, half:]
    parts = [tl, bl, tr, br]
    return parts

def Quadsplit(img, lvl):
    tree, tmp = [[img]], [img]
    for i in range(lvl):
        tmp2 = []
        for j in range(len(tmp)):
            parts = Two_Dim_Image_Split(tmp[j])
            for p in parts:
                tmp2.append(p)
        tmp = tmp2
        tree.append(tmp)
    return tree

def DisplayLevel(tree, lvl):
    for i in tree[lvl]:
        cv2.imshow(random_string(7), i)
    print('Size of Level: ', len(tree[lvl]))

img = cv2.imread('airport.jpg')
lvl = int(input('Enter Quadtree Build Level: '))
tree = Quadsplit(img, lvl)
lvl = int(input('Enter Quadtree Display Level: '))
DisplayLevel(tree, lvl)
cv2.waitKey(0)