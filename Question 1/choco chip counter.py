def countchoco(image):
    gray_im = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    plt.subplot(221)
    plt.title('Grayscale image')
    plt.imshow(gray_im, cmap="gray", vmin=0, vmax=255)

    # Contrast adjusting with gamma correction y = 1.2

    gray_correct = np.array(255 * (gray_im / 255) ** .07 , dtype='uint8')
    plt.subplot(222)
    plt.title('Gamma Correction')
    plt.imshow(gray_correct, cmap="gray", vmin=0, vmax=255)
    # Contrast adjusting with histogramm equalization
    gray_equ = cv2.equalizeHist(gray_im)
    plt.subplot(223)
    plt.title('Histogram equilization')
    plt.imshow(gray_correct, cmap="gray", vmin=0, vmax=255)
    plt.show()

    # Local adaptative threshold

    lower = np.array([140])  #-- Lower range --
    upper = np.array([256])  #-- Upper range --
    mask = cv2.inRange(gray_correct, lower, upper)
    res = cv2.bitwise_and(gray_correct, gray_correct, mask= mask)  #-- Contains pixels having the gray color--
    cv2.imshow('Result', res)

    gray_correct = res

    thresh = cv2.adaptiveThreshold(gray_correct, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 20)
    plt.title('Local adapatative Threshold')
    plt.imshow(thresh, cmap="gray", vmin=0, vmax=255)
    plt.show()

    # Dilatation et erosion
    kernel = np.ones((1, 4), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    img_erode = cv2.erode(img_dilation, kernel, iterations=1)
    # clean all noise after dilatation and erosion
    img_erode = cv2.medianBlur(img_erode, 1)
    plt.subplot(221)
    plt.title('Dilatation + erosion')
    plt.imshow(img_erode, cmap="gray", vmin=0, vmax=255)

    img_not = cv2.bitwise_not(img_erode)
    img_not = cv2.GaussianBlur(img_not,(5,5),3)
    cv2.imshow("Invert1", img_not)

    img_erode = img_not

    plt.show()

    # Labeling

    ret, labels = cv2.connectedComponents(img_erode)
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    plt.subplot(222)
    plt.title('Objects counted:'+ str(ret-1))
    plt.imshow(labeled_img)
    print('objects number is:', ret-1)
    plt.show()
