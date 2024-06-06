
#안쓰임
import cv2
import numpy as np
from sklearn.cluster import KMeans
#쓰임
import struct
import random
import matplotlib.pyplot as plt


#이미지 선언 파트
class ImageData:
    def __init__(self, data, height, width):
        self.data = data
        self.height = height
        self.width = width

# BMP 파일을 불러오는 함수
def readBmp(file_path) -> ImageData:
    with open(file_path, 'rb') as f:
        header = f.read(54)  # BMP 파일 헤더는 보통 54바이트입니다.
        width = struct.unpack('<i', header[18:22])[0]
        height = struct.unpack('<i', header[22:26])[0]
        data = []
        for _ in range(height):
            for _ in range(width):
                b, g, r = struct.unpack('<BBB', f.read(3))
                data.append((r, g, b))  # BMP 파일은 BGR 순서입니다.
    return ImageData(data=data, height=height, width=width)


# 초기 클러스터 중심을 선택하는 함수
def initialize_centroids_kmeans_plusplus(imageData: ImageData, k: int) -> list:
    centroids = []  # 센트로이드 리스트
    
    # 첫 번째 센트로이드는 무작위로 선택
    first_centroid_idx = random.randint(0, len(imageData.data) - 1)
    centroids.append(imageData.data[first_centroid_idx])
    
    # 나머지 센트로이드 선택
    for _ in range(1, k):
        # 각 점까지의 최소 거리의 제곱 계산
        min_distances = [min(sum((point[i] - centroid[i]) ** 2 for i in range(3)) 
                             for centroid in centroids) for point in imageData.data]
        # 새로운 센트로이드 선택 확률 계산
        probabilities = [distance / sum(min_distances) for distance in min_distances]
        # 새로운 센트로이드 선택
        new_centroid_idx = random.choices(range(len(imageData.data)), weights=probabilities)[0]
        centroids.append(imageData.data[new_centroid_idx])
    
    return centroids

# 초기 클러스터 중심을 무작위로 선택하는 함수
def initialize_centroids(imageData: ImageData, k: int) -> list:
    centroids = []  # 센트로이드 리스트
    chosen = set()  # 이미 선택된 클러스터 중심의 인덱스를 저장하기 위한 세트
    flattened_data_length = imageData.height * imageData.width  # 1차원 리스트 길이
    while len(centroids) < k:
        idx = random.randint(0, flattened_data_length - 1)
        if idx not in chosen:
            centroids.append(imageData.data[idx])
            chosen.add(idx)
    return centroids

# 유클리디안 거리 계산 및 클러스터 할당 함수
def assign_clusters(imageData, centroids):
    clusters = [] #클러스터스
    for x in imageData: #모든 이미지 픽셀을 순회한다.
        distances = [sum((x[i] - c[i]) ** 2 for i in range(len(x))) for c in centroids]
        #클러스터 거리를 계산한다. 모든 랜덤으로 선택된 클러스터는 x라는 한 픽셀과 거리를 계산한다. 
        #len(x)는 그 배열의 길이를 반환하는 함수다 즉 3번 반복된다
        #x - c는 x와 y의 차. 즉 거리를 계산하는 과정이다. 길이 넓이의 각 제곱의 합은 곧 거리값과 비례한다.
        cluster = distances.index(min(distances)) #최종적으로 가장 거리가 짧은 값을 클러스터로 선정한다.
        clusters.append(cluster) #배열에 하나씩 저장한다. 모든 데이터는 1차원 배열로 저장되며 형태적으로 이미지 데이터와 같다.
    return clusters

# 클러스터 중심 재계산 함수
def update_centroids(imageData, clusters, centroids, k):
    new_centroids = [] #새로운 중심값 넣는 배열
    for i in range(k): #k 번 반복해야 한다.
        cluster_points = [imageData[j] for j in range(len(imageData)) if clusters[j] == i] #i번째 클러스터를 지목하고 있는 모든 픽셀을 뽑고.
        if not cluster_points: #클러스터 포인트가 존재하지 않는다면
            new_centroids.append(centroids[i]) #그냥 원래대로의 클러스터 값을 넣는다.
            continue #일반적인 예외처리 과정이다. 거의 시행되지 않는다.
        new_centroid = [sum(pixel[j] for pixel in cluster_points) / len(cluster_points) for j in range(len(cluster_points[0]))]
        # 클러스터에 존재하는 모든 픽셀의 값(rgb값 싸그리 합친다)들을 합치고 클러스터 개수만큼 나눈다.
        # 그러면 모든 rgb값의 평균이 도출된다 이게 새로운 클러스터 값(rgb색상형태)이고. 이것을 배열애 넣는다.
        new_centroids.append(new_centroid) #그럼 클러스터 번호 순서대로 저장된다.
    return new_centroids

def calculate_distance(point1, point2):
    # 각 차원의 제곱을 계산하고 합산한 후 제곱근을 취하여 거리를 구함
    squared_distance = sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2))
    distance = squared_distance ** 0.5
    return distance

# K-means 알고리즘 수행 함수
def kmeans(imageData: ImageData, k: int, max_iterations: int = 25, tolerance : float= 0.1):
    data = imageData.data
    centroids = initialize_centroids(imageData, k) #초기 중심값을 설정함
    for _ in range(max_iterations): #반복한다. 이때 중요한 점은 int는 미리 설정해둔 반복 횟수다. 반복 횟수만큼 반복한다.
        clusters = assign_clusters(data, centroids) #할당된 클러스터를 별도의 배열에 저장하는 과정 
        new_centroids = update_centroids(data, clusters, centroids, k) 
        #centorids는 초기 클러스터 값이다. 
        #clusters는 데이터 포인트들의 그룹을 나타낸다.
        if max(calculate_distance(new_centroid, centroid) for new_centroid, centroid in zip(new_centroids, centroids)) < tolerance:
            break
        centroids = new_centroids #새로운 클러스터를 다시 배열한다.
    return centroids, clusters #각 중심값을 표현하는 배열과 모든 데이터들이 어떤 클러스터에 할당되었는지 표현하는 함수

#최종적 이미지 매핑
def segment_image(image_data: ImageData, centroids, clusters):
    segmented_image = []
    idx = 0
    for i in range(image_data.height):
        row = []
        for j in range(image_data.width):
            row.append(tuple(int(c) for c in centroids[clusters[idx]])) 
            #clusters는 데이터 군집이다. 새로운 k값의 인덱스가 들어있다.
            #centroids k값 rgb값이다.
            #요약하면 clusters의 한 데이터는 어떤 k를 적용할 지에 대한 내용이 들어 있다.
            #centroids[clusters[idx]]의 c를 튜플로 만들어서 3개의 채널의 하나의 배열로 만든다.
            idx += 1 #idx값을 1증가시킨다. 반복문의 i(j)의 역할을 할 것이다.
        segmented_image.append(row)
    return segmented_image

#뒤집기
def flip_image_upside_down(image_data):
    return image_data[::-1]

# 분할된 이미지를 시각화하는 함수 (넘파이 없이 뒤집힌 이미지 사용)
def show_segmented_image(image_data):
    flipped_image = flip_image_upside_down(image_data)
    plt.imshow(flipped_image)
    plt.axis('off')
    plt.show()

# 사용 예시
a = readBmp("i.bmp")
k = 2  # 클러스터 수
centroids, clusters = kmeans(a, k)
segmented_image = segment_image(a, centroids, clusters)
show_segmented_image(segmented_image)

# 이미지를 불러오고 데이터를 준비합니다.
# image = cv2.imread("i.bmp")
# data = image.reshape((-1, 3)).astype(np.float32)

# # K-means 클러스터링을 수행합니다.
# k = 5  # 클러스터 수
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# # 각 픽셀에 클러스터 중심값을 할당합니다.
# centers = np.uint8(centers)
# segmented_data = centers[labels.flatten()]
# segmented_image = segmented_data.reshape(image.shape)

# # 결과를 표시합니다.
# cv2.imshow("Segmented Image", segmented_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
