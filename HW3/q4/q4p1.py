import cv2
import numpy as np
import glob
import os
import random
from sklearn.neighbors import KNeighborsClassifier

INPUT_PATH = "/home/sahel/university/vision/HW3/Data/Train/*/"
# INPUT_PATH = "/home/sahel/university/vision/HW3/Data/Train/Mountain/*.jpg"
directories = glob.glob(INPUT_PATH)


def read_data(directories):
    all_images = [[], [], [], [], []]
    image_category = []
    for path in directories:
        img_path_regex = os.path.join(path, "*.jpg")
        image_paths = glob.glob(img_path_regex)
        random.shuffle(image_paths)
        # current_images = []
        image_category.append(path.split('/')[-2])
        parts = int(len(image_paths)/5)
        for i in range(5):
            current_images = []
            for j in range(parts):
                current_index = i*j
                path = image_paths[current_index]
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                current_images.append(image)
            all_images[i].append(current_images)
    print(len(all_images), len(all_images[0]))
    return all_images, image_category


def read_test(directories):
    images = []
    image_category = []
    for path in directories:
        img_path_regex = os.path.join(path, "*.jpg")
        image_paths = glob.glob(img_path_regex)
        random.shuffle(image_paths)
        current_images = []
        image_category.append(path.split('/')[-2])
        for image_path in image_paths:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            current_images.append(image)
        images.append(current_images)
    return images, image_category


# def compute_score(knn, validation_x, validation_y):
#     total_correct = 0
#     predicted_ys = knn.predict(validation_x)
#     for i, y in enumerate(predicted_ys):
#         if y == validation_y[i]:
#             total_correct += 1
#     validation = (total_correct/len(validation_x))*100
#     return validation


def train_data(k, image_size, validation_index):
    print(validation_index)
    # print("image size : ", image_size)
    x = []
    y = []
    validation_x = []
    validation_y = []
    for j in range(5):
        if j == validation_index:
            for i, category in enumerate(image_category):
                # print(i, category)
                category_images = all_images[j][i]
                for image in category_images:
                    resized = cv2.resize(image, image_size)
                    feature_vector = resized.reshape((-1))
                    validation_x.append(feature_vector)
                    validation_y.append(i)

        else:
            for i, category in enumerate(image_category):
                # print(i, category)
                category_images = all_images[j][i]
                for image in category_images:
                    resized = cv2.resize(image, image_size)
                    feature_vector = resized.reshape((-1))
                    x.append(feature_vector)
                    y.append(i)
    knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    knn.fit(x, y)
    print("train len : ", len(x))
    print("validation len : ", len(validation_x))
    score = knn.score(validation_x, validation_y)
    return score, knn


def tune_hyper_parameter():
    # image_sizes = [8, 16, 24, 32]
    image_sizes = [5, 10, 15, 20, 25, 30, 35]
    max_k = 5
    best_score = -1
    best_knn = None
    hyper_parameter = None
    for validation_index in range(1, 5):
        for k in range(1, max_k+1):
            for image_size in image_sizes:
                score, knn = train_data(k, (image_size, image_size), validation_index)
                print("finished train with score : ", score, " k : ", k, " image size : ", image_size)
                if score > best_score:
                    best_score = score
                    best_knn = knn
                    hyper_parameter = (validation_index, k, image_size)
    test_x, test_y = ready_test_data(hyper_parameter[2])
    print("final score : ", best_knn.score(test_x, test_y))


def ready_test_data(image_size):
    x = []
    y = []
    for i, category in enumerate(test_category):
        category_images = test_images[i]
        for image in category_images:
            resized = cv2.resize(image, (image_size, image_size))
            feature_vector = resized.reshape((-1))
            x.append(feature_vector)
            y.append(i)
    return x, y


all_images, image_category = read_data(directories)
test_images, test_category = read_test(glob.glob("/home/sahel/university/vision/HW3/Data/Test/*/"))
tune_hyper_parameter()

print(len(all_images), len(image_category))
print(image_category)



