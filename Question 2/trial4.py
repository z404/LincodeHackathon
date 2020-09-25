import cv2

img = cv2.imread('test5.jpg')#,0)
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50,150)
    cv2.imshow("lol", canny)
canny(img)
