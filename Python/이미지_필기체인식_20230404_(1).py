# import cv2_imshow
import cv2
import joblib
import numpy as np

# Load the classifier
clf = joblib.load("digits_cls.pkl")

# Read the input image
im = cv2.imread("photo_1.jpg")

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)  # for image smoothing

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)  # 이진화

# Find contours in the image
ctrs, _ = cv2.findContours(
    im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# For each rectangular region, calculate HOG features and predict the digit using Linear SVM.
for rect in rects:
    # Draw the rectangles
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] +
                  rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))  # 팽창
    nbr = clf.predict(np.array([roi], 'float64').reshape(-1, 784))
    cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),
                cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

# Display image with output text
# cv2_imshow(im)
cv2.imshow(im)
cv2.waitKey(0)
cv2.destroyAllWindows()
