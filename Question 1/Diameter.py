from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
import math
from colormap import rgb2hex
from sklearn.cluster import KMeans

def circle_equation(x,y,r,l1,image):
   l2=["",""]
   r=r-17
   l2[0] = int((l1[0]+l1[2])/2) #x centre of circle
   l2[1] = int((l1[1]+l1[3])/2) #y centre of circle
   img = cv2.circle(image,(l2[0],l2[1]),int(r/2),(255, 0, 0),1)
   #plt.imshow(img)
   if (((x-l2[0])**2 + (y-l2[1])**2)>r**2):
      return 0
   elif(((x-l2[0])**2 + (y-l2[1])**2) <= r**2):
      return 1

def circle_check(x,y):
   return (circle_equation(x,y,dA,l1,image))

def hex_cookie(image,l):
   crop_img = image[l[0][1]:l[3][1], l[3][0]:l[2][0]]
   #cv2.imwrite("crop.png",crop_img)
   dict1 = {}

   def make_histogram(cluster):
      numLabels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
      hist, _ = np.histogram(cluster.labels_, bins=numLabels)
      hist = hist.astype('float32')
      hist /= hist.sum()
      return hist

   def make_bar(height, width, color):
      
      bar = np.zeros((height, width, 3), np.uint8)
      bar[:] = color
      red, green, blue = int(color[2]), int(color[1]), int(color[0])
      hsv_bar = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
      hue, sat, val = hsv_bar[0][0]
      return bar, (red, green, blue), (hue, sat, val)

   def sort_hsvs(hsv_list):
      
      bars_with_indexes = []
      for index, hsv_val in enumerate(hsv_list):
         bars_with_indexes.append((index, hsv_val[0], hsv_val[1], hsv_val[2]))
      bars_with_indexes.sort(key=lambda elem: (elem[1], elem[2], elem[3]))
      return [item[0] for item in bars_with_indexes]

   height, width, _ = np.shape(crop_img)

   # reshape the image to be a simple list of RGB pixels
   image = crop_img.reshape((height * width, 3))

   # we'll pick the 5 most common colors
   num_clusters = 5
   clusters = KMeans(n_clusters=num_clusters)
   clusters.fit(image)

   # count the dominant colors and put them in "buckets"
   histogram = make_histogram(clusters)
   # then sort them, most-common first
   combined = zip(histogram, clusters.cluster_centers_)
   combined = sorted(combined, key=lambda x: x[0], reverse=True)

   # finally, we'll output a graphic showing the colors in order
   bars = []
   hsv_values = []

   top3 = []
   
   for index, rows in enumerate(combined):
      bar, rgb, hsv = make_bar(100, 100, rows[1])
      #print(f'Bar {index + 1}')
      #print(f'  RGB values: {rgb}')
      #print(f'  HSV values: {hsv}')

      hsv_values.append(hsv)
      bars.append(bar)

      if not(rgb[0]>150 and rgb[1]>150 and rgb[1]>150):
          top3.append(rgb)
          if len(top3) == 3:
              break

   #print(top3)
   
   # sort the bars[] list so that we can show the colored boxes sorted
   # by their HSV values -- sort by hue, then saturation
   #sorted_bar_indexes = sort_hsvs(hsv_values)
   #sorted_bars = [bars[idx] for idx in sorted_bar_indexes]

   #cv2.imshow('Sorted by HSV values', np.hstack(sorted_bars))
   #cv2.imshow(f'{num_clusters} Most Common Colors', np.hstack(bars))
   #cv2.waitKey(0)
   top3hex = []
   
   for i in top3:
       r,g,b = i
       top3hex.append(rgb2hex(r,g,b))
   
   for i in range (0,crop_img.shape[0]):
      for j in range(0,crop_img.shape[1]):
         dict1[(i,j)] = crop_img[i,j]
   '''
   for i in range(0,crop_img.shape[0]):
      for j in range (0,crop_img.shape[1]):
         if circle_check(i,j)==1:
            r,g,b = dict1[i,j]
            print (r,g,b)
            print(rgb2hex(dict1[i,j][0],dict1[i,j][1],dict1[i,j][2]), i,j)
   '''
   return top3hex

def finddandhex(image):

    def midpoint(ptA, ptB):
       return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


    def canny(image):
       gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       blur = cv2.GaussianBlur(gray, (5, 5), 0)
       canny = cv2.Canny(blur, 50,150)
       return canny

    def region_interest(image):
       height = image.shape[0]
       polygons = np.array([
          [(10, 200), (400,200), (400,10), (10,10)]
          ])
       mask = np.zeros_like(image)
       cv2.fillPoly(mask,polygons,255)
       masked_image = cv2.bitwise_and(image,mask)
       #cv2.imshow("ay",mask)
       return masked_image

    #pixel 0.0264583333

    
    #print(image.shape)
    #cv2.imshow("ayy", image)
    image2 = np.copy(image)
    canny = canny(image2)
    cropped_image = region_interest(canny)
    #cv2.imshow("lol", region_interest(canny))
    #plt.imshow(canny)
    #plt.show()
    #cv2.waitKey(0)

    width = 8.1
    #(5cm , 2inch good day), (boroline 5.4cm, 2.12598inch), at 25.5 cm height tripod
    ratio = width/image.shape[0]

    edged = cv2.dilate(cropped_image, None, iterations=5)
    edged = cv2.erode(edged, None, iterations=0)
    #cv2.imshow("hio",edged)

    # find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    l=[]
    for c in cnts:
        if cv2.contourArea(c) < 2000:
            continue
        # compute the rotated bounding box of the contour
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        # loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
            #print(int(x), int(y))
            l.append([int(x), int(y)])

        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        # draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)

        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        
        # if the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to supplied metric
        # (in this case, inches)
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / ratio
        #dimA = dA / pixelsPerMetric
        #dimB = dB / pixelsPerMetric

        #print (dimA,"dimA") #width
        #print(dimB,"dimB") #height

        #cv2.putText(orig, "{:.1f}in".format(dA),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
        #cv2.putText(orig, "{:.1f}in".format(dB),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)

        #distace(l)
        #cv2.imshow("Image", orig)
        #cv2.waitKey(0)
            
        tophex = hex_cookie(image,l)
    
        return dB*ratio,dA*ratio,tophex

def countchoco(original):
    gray_im = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    #plt.subplot(221)
    #plt.title('Grayscale image')
    #plt.imshow(gray_im, cmap="gray", vmin=0, vmax=255)

    # Contrast adjusting with gamma correction y = 1.2
    gray_correct = np.array(255 * (gray_im / 255) ** .07 , dtype='uint8')
    #plt.subplot(222)
    #plt.title('Gamma Correction')
    #plt.imshow(gray_correct, cmap="gray", vmin=0, vmax=255)
    
    # Contrast adjusting with histogramm equalization
    gray_equ = cv2.equalizeHist(gray_im)
    #plt.subplot(223)
    #plt.title('Histogram equilization')
    #plt.imshow(gray_correct, cmap="gray", vmin=0, vmax=255)
    #plt.show()

    # Local adaptative threshold

    lower = np.array([140])  #-- Lower range --
    upper = np.array([256])  #-- Upper range --
    mask = cv2.inRange(gray_correct, lower, upper)
    res = cv2.bitwise_and(gray_correct, gray_correct, mask= mask)  #-- Contains pixels having the gray color--
    #cv2.imshow('Result', res)

    gray_correct = res

    thresh = cv2.adaptiveThreshold(gray_correct, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 20)
    #plt.title('Local adapatative Threshold')
    #plt.imshow(thresh, cmap="gray", vmin=0, vmax=255)
    #plt.show()

    # Dilatation et erosion
    kernel = np.ones((1, 4), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    img_erode = cv2.erode(img_dilation, kernel, iterations=1)
    
    # clean all noise after dilatation and erosion
    img_erode = cv2.medianBlur(img_erode, 1)
    #plt.subplot(221)
    #plt.title('Dilatation + erosion')
    #plt.imshow(img_erode, cmap="gray", vmin=0, vmax=255)

    img_not = cv2.bitwise_not(img_erode)
    img_not = cv2.GaussianBlur(img_not,(5,5),3)
    #cv2.imshow("Invert1", img_not)

    img_erode = img_not

    #plt.show()

    # Labeling

    ret, labels = cv2.connectedComponents(img_erode)
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    #plt.subplot(222)
    #plt.title('Objects counted:'+ str(ret-1))
    #plt.imshow(labeled_img)
    #plt.show()

    return ret-1

image = cv2.imread("4.jpg")
image = cv2.resize(image, None, fx=0.05, fy=0.05, interpolation=cv2.INTER_CUBIC)

diametertuple = finddandhex(image)

print("Horisontal Diameter =",diametertuple[0])
print("Vertical Diameter =",diametertuple[1])
print("Number of chocolate chips:",countchoco(image))
print("Top 3 Hex codes:",diametertuple[2])
