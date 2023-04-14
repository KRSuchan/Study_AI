# from google.colab.patches import cv2_imshow
import cv2  # cv2 대신에 opencv_python 을 설치 (제대로 설치 안되면 python 도 최신 버전으로 재설치)
import os

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(
    cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
face_detector = cv2.CascadeClassifier(haar_model)

img = cv2.imread("face_img.jpg")
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rects = face_detector.detectMultiScale(grayImg)
print(rects)

for rect in rects:
    cv2.rectangle(img, (rect[0], rect[1]), (rect[0] +
                  rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
cv2.imshow(img)
