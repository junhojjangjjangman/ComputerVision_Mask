import cv2
import numpy as np
import sys
import Dir
import pandas as pd
import matplotlib.pyplot as plt

img = cv2.imread(Dir.dir+"[Dataset] Module 20 images/image001.png")
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Covert_img = cv2.imread(Dir.dir+"[Dataset] Module 20 images/image001.png", 0)

ret,thresholded = cv2.threshold(grey,29,255,cv2.THRESH_BINARY_INV)    #we use cv2.THRESH_BINARY_INV instead of cv2.THRESH_BINARY
masked = cv2.bitwise_and(img, img, mask = thresholded)
cv2.imshow("Masked", masked)

cv2.waitKey(0)                            # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

mask = img.copy()                         # 우리가 만들 마스크 이미지. 초기 이미지의 복사본으로 초기화합니다.
(b,g,r) = cv2.split(img)                  # BGR 이미지를  각 채널별로 분할하여 별도로 작업할 수 있습니다.
mask[(b==255)&(g==255)&(r==255)] = 0     # 흰색 배경(BGR 채널이 모두 255인 경우)을 0(검정색)으로 변경합니다.

cv2.imshow("Mask",mask)

cv2.waitKey(0)                            # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

mask_inv = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
plt.imshow(mask_inv)
plt.show()

cv2.imshow("Blue Mask",mask[:,:,0])       # 단어가 어떻게 파란색인지 주목하십시오.
cv2.imshow("Green Mask",mask[:,:,1])
cv2.imshow("Red Mask",mask[:,:,2])
cv2.waitKey(0)                            # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

mask[300:,:,1]=0                          # 이미지는 행렬임을 기억하십시오. 아래쪽 절반을 검은색(0)으로 만듭니다.
cv2.imshow("Green Mask",mask[:,:,1])
cv2.waitKey(0)                           # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

mask[400:,:,2]=0                          # 이미지는 행렬임을 기억하십시오. 아래쪽 절반을 검은색(0)으로 만듭니다.
cv2.imshow("Red Mask",mask[:,:,2])
cv2.waitKey(0)                           # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

mask[300:,:,0]=0                          # 이미지는 행렬임을 기억하십시오. 아래쪽 절반을 검은색(0)으로 만듭니다.
cv2.imshow("Blue Mask",mask[:,:,0])
cv2.waitKey(0)                           # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

mask[300:,:,2]=0                          # 이미지는 행렬임을 기억하십시오. 아래쪽 절반을 검은색(0)으로 만듭니다.
cv2.imshow("Blue Mask",mask[:,:,2])
cv2.waitKey(0)                           # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()
# 실행 시 글자 부분에 노이즈가 발생함 파랑으로 보이지만 Red, Green이 껴 있음
# 노이즈란?
# 원하는 신호의 전송 및 처리를 방해하는 `원치않는 파형`

# 잡음의 특징
# 정보를 포함하고 있지 않는 신호의 일종
# 통상, 유용한 정보 신호에 더해져서(부가되어,additive) 나타남
# 신호의 존재 유무와 상관없이 인공이든 자연적이든 거의항상 존재하는 경향이 있음

# 노이즈 발생 시 블러링 기법을 사용하여 노이즈에 영향을 안 받게 할 수 있음
# 가우시안 블러

# 이미지에서의 노이즈
# 이미지도 음성 신호처럼 주파수로 표현할 수 있다. 일반적으로 고주파는 밝기 변화가 많은 곳, 즉 경계선(edge) 영역에서 나타나며,
# 저주파는 밝기 변화가 적은 곳인 배경의 영역에 주로 나타난다.
# 이를 이용해 고주파를 제거하면 경계선의 변화가 약해져 Blur 처리가 되고,
# 저주파를 제거하면 배경 영역이 흐려져 대상의 영역(edge)를 뚜렷하게 확인할 수 있다.
# 이미지 필터링은 kernel(filter)라고 하는 정방행렬을 정의하고,
# 이 커널을 이미지 위에서 이동시켜가면서 커널과 겹쳐진 이미지 영역과 연산한 후 그 결과값을 연산을 진행한 이미지 픽셀을 대신하여 새로운 이미지를 만드는 연산이다.

# openCV에서는
# 이미지와 kernel(filter)를 Convolution(합성곱)하여 이미지를 필터링 해주는 함수, cv2.filter2D를 제공
# cv2.filter2D(src, ddepth, kernel [, dst [, anchor [, delta [, borderType]]]]) → dst
# 1. src : 입력 이미지
# 2. ddepth : 출력 이미지에 적용되는 이미지의 깊이 (자료형 크기), -1이면 입력과 동일하게 적용된다. 입력이미지와 출력이미지의 조합을 확인해보고 싶으면 Image Filtering에서 depth combination을 확인해보면 된다.
# 3. kernel : 합성곱 kernel, 해당 kernel은 float 상수로 구성된 단일 채널이어야 한다. 다른 채널에 각각 다른 커널을 적용하고 싶다면 split 함수를 이용해 해당 이미지의 채널을 분리한 뒤 개별적으로 적용해야 한다.
# 4. anchor : 커널 내에서 필터링된 지점의 상대적 위치를 나타내는 커널의 앵커(닻), 즉 필터링할 이미지 픽셀의 위치가 커널의 어디에 존재해야 하는지 그 기준점을 지정해준다. default = (-1, -1)로 앵커가 커널의 중심에 있음을 의미한다.
# 5. delta : dst에 저장하기 전에 필터링된 픽셀에 추가적으로 더해주는 값 (bias의 역할)

# 필터링하는 방법
# 1. Average Filter
# K = 1 / 25 * [ 1, 1, 1, 1, 1 ]
#              [ 1, 1, 1, 1, 1 ]
#              [ 1, 1, 1, 1, 1 ]
#              [ 1, 1, 1, 1, 1 ]
#              [ 1, 1, 1, 1, 1 ]
# 모든 Filtering 커널 행렬의 전체 원소의 합은 1이 되도록 만들어지기 때문에 총 원소의 합(25)으로 나누어 정규화한다.
#
# 해당 커널이 적용되는 방법은 다음과 같다.
# 1. 픽셀을 중심으로 5 * 5의 영역을 만든다.
# 2. 이 영역에 속하는 픽셀 값과 커널을 곱하여 그 값을 합친다.
# 3. 더한 값을 25로 나누고 이 값을 적용한 픽셀 값으로 초기화 한다.
# 이러한 계산방법 덕분에 커널이 커지면 커질수록 이미지는 더더욱 흐려진다.

# 이미지 블러링 (Image Blurring)
# 1. Averaging --> cv2.blur(src, ksize [, dst [, anchor [, borderType]]]) → dst
# 가장 일반적인 필터링 방법
# src : 입력 이미지, 채널 수는 상관 없으나 다음과 같은 데이터 타입에만 사용할 수 있다.
# (CV_8U, CV_16U, CV_16S, CV_32F, CV_64F)
# ksize : 커널의 크기 입력 이미지와 커널의 크기 등을 인자로 전달하면,
# blur 함수 내에서 정규화된 커널을 만들고 해당 커널을 입력 이미지에 적용하여 블러 처리된 이미지를 출력한다.

# 2. Gaussian Filtering --> cv2.GaussianBlur(src, ksize, sigmaX [, dst [, sigmaY [, borderType]]]) → dst
# Gaussian Noise (전체적으로 밀도가 동일한 노이즈, 백색노이즈 등)를 제거하는데 가장 효과적
# src : 입력 이미지. 채널 수는 상관 없으나 다음과 같은 데이터 타입에만 사용할 수 있다.
# (CV_8U, CV_16U, CV_16S, CV_32F, CV_64F)
# ksize : 커널의 크기. GaussianBlur를 적용하기 위한 커널의 크기는 반드시 양수의 홀수이어야 한다.
# sigmaX : 가우시안 커널의 X방향 표준편차
# sigmaY : 가우시안 커널의 Y방향 표준편차
# (sigmaY = 0이면 sigmaX와 같은 크기가 적용된다. 만약 X와 Y 둘 모두 크기가 0이라면 커널의 너비와 높이에 따라 계산된다.)

# 3. Median Filtering --> medianBlur(src, ksize [, dst]) → dst
# 점 잡음(salt-and-pepper noise) 제거에 효과적
# src : 입력 이미지, 채널 수가 1, 3, 4개여야 한다. 또한 이미지 타입은 CV_8U여야 한다.
# ksize : 커널의 크기. MedianBlur를 적용하기 위한 커널의 크기는 반드시 1보다 큰 홀수여야 한다.

# 4. Bilateral Filtering --> cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace [, dst [, borderType]]) → dst
# 경계선을 유지하면서 Gaussian Filtering이 하는 것처럼 Blur 처리를 해주는 방법
# src : 8 bit, 채널 수가 1, 3인 이미지
# d : 커널의 크기, 필터링 시 고려할 주변 픽셀의 지름. 양수가 아니면 sigmaSpace로 계산된다.
# sigmaColor : 색공간 표준편차. 값이 크면 색이 많이 달라도 서로 영향을 미친다.
# sigmaSpace : 거리공간 표준편차. 값이 크면 멀리 떨어져 있는 픽셀들이 서로 영향을 미친다. d가 양수이면 sigmaSpace에 상관없이 인접한 픽셀들에 영향을 미치지만, 그렇지 않으면 d는 sigmaSpace에 비례한다.

img = cv2.imread(Dir.dir+'img.jpg')
# 커널사이즈 : 양수에 홀수 - (3, 3), (5, 5), (7, 7)이런 사이즈에 우수한 성능을 보여줌
kernel_3 = cv2.GaussianBlur(img, (3,3), 0) # 0: 시그마 x 표준편차
kernel_5 = cv2.GaussianBlur(img, (5,5), 0)
kernel_7 = cv2.GaussianBlur(img, (7,7), 0)
cv2.imshow('img', img)
cv2.imshow('kernel_3', kernel_3)
cv2.imshow('kernel_5', kernel_5)
cv2.imshow('kernel_7', kernel_7)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread(Dir.dir+"[Dataset] Module 20 images/image001.png")
print(img.shape)
# cv.resize(img,dsize,fx,fy,interpolation)
# img - Image
# dsize - Manual Size, 가로, 세로 형태의 tuple
# fx - 가로 사이즈의 배수, 2배로 크개하려면 2. 반으로 줄이려면 0.5
# fy - 세로 사이즈의 배수(종횡비를 유지하기 위해 dsize보다 fx,fy를 많이 사용)
# inter polation - 보간법
#   INTER_NEAREST
#   INTER_LINEAR(크기 늘릴 때 사용)
#   INTER_AREA(크기 줄일 때 사용)
#   INTER_CUBIC(크기 늘일 때 사용, 속도 느림, 퀄리티 좋음)
#   INTER_LANCZOS4 등
resized = cv2.resize(img,(400, 300))           # 두 번째 매개변수는 원하는 모양(너비, 높이)입니다.
cv2.imshow("img",img)
cv2.imshow("Resized",resized)
cv2.waitKey(0)                                 # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()


resized = cv2.resize(img,(200, 300))           # 두 번째 매개변수는 원하는 모양(너비, 높이)입니다.
cv2.imshow("img",img)
cv2.imshow("Resized",resized)
cv2.waitKey(0)                                 # 아무 키나 누른 후 창 종료
cv2.destroyAllWindows()

img = cv2.imread(Dir.dir+'img.jpg')
dst = cv2.resize(img, None, fx=2, fy = 2)  # x, y비율 정의(2배로 축소)
cv2.imshow('img', img)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread(Dir.dir+'img.jpg')
dst = cv2.resize(img, None, fx=1.5, fy = 1.5 , interpolation = cv2.INTER_CUBIC )  # x, y비율 정의(0.5배로 축소)
cv2.imshow('img', img)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

h,w,c = img.shape
# 이미지를 잘라서 위쪽 절반만 얻기:
cv2.imshow("img",img)
cv2.imshow("Cropped Top",img[:h//2,:,:])
cv2.waitKey(0)
cv2.destroyAllWindows()

# 이미지를 잘라서 오른쪽 반만 얻기
cv2.imshow("img",img)
cv2.imshow("Cropped Top",img[:,w//2:,:])
cv2.waitKey(0)
cv2.destroyAllWindows()

# 이미지 합성 일부 잘라서 원본 데이터에 합성하기
img = cv2.imread(Dir.dir+'img.jpg')
crop = img[100:200, 200:400] # 세로기준 100:200, 가로기준 300:400
img[100:200, 400:600] = crop
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()