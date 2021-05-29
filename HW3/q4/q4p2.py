import json

import cv2
import numpy as np
import glob
import os
import random

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
INPUT_PATH = "/home/sahel/university/vision/HW3/Data/Train/*/"
categories = ['Store', 'Street', 'Open_Country', 'Kitchen', 'Coast', 'Highway',
              'Forest', 'Livingroom', 'Bedroom', 'Industrial', 'Suburb', 'Office',
              'Mountain', 'Tall_Building', 'Inside_City']


def read_data(input_path):
    directories = glob.glob(input_path)
    data = []
    for i in range(len(categories)):
        data.append([])
    for directory_path in directories:
        current_images = []
        name = directory_path.split('/')[-2]
        category = categories.index(name)
        img_path_regex = os.path.join(directory_path, "*.jpg")
        image_paths = glob.glob(img_path_regex)
        for image_path in image_paths:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # data.append([image, category])
            current_images.append(image)
        data[category].extend(current_images)
    return data


def extract_image_features(data):
    features = []
    sift = cv2.SIFT_create()
    for i, category_images in enumerate(data):
        category_features = []
        for image in category_images:
            kp1, des1 = sift.detectAndCompute(image, None)
            category_features.append(des1.tolist())
        features.append(category_features)
    return features


def create_visual_words(features, cluster_num):
    flat_features = [descriptor for category in features for feature_list in category for descriptor in feature_list]
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(flat_features)
    visual_words = kmeans.cluster_centers_
    return visual_words


def create_image_histogram(visual_words, features):
    knn = KNeighborsClassifier(n_neighbors=int(len(visual_words)/10), metric='euclidean')
    knn.fit(visual_words, list(range(len(visual_words))))
    image_histograms = []
    for category_images in features:
        category_histograms = []
        for image_descriptors in category_images:
            probs = knn.predict_proba(image_descriptors)
            histogram = probs.sum(axis=0)
            category_histograms.append(histogram.tolist())
        image_histograms.append(category_histograms)
    return image_histograms


def train_knn(train_histograms, k, validation_histograms):
    knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    x = []
    y = []
    validation_xs = []
    validation_ys = []
    for i, train_category in enumerate(train_histograms):
        train_y = np.repeat(i, len(train_category))
        x.extend(train_category)
        y.extend(train_y.tolist())
    knn.fit(x, y)
    for i, validation_category in enumerate(validation_histograms):
        validation_y = np.repeat(i, len(validation_category))
        validation_xs.extend(validation_category)
        validation_ys.extend(validation_y.tolist())
    score = knn.score(validation_xs, validation_ys)
    return score, knn

#
# def test_code():
#     data = read_data(INPUT_PATH)
#     print("read data finished")
#     features = extract_image_features(data)
#     data = None
#     print("extracted features")
#     visual_words = create_visual_words(features, 50)
#     print("get visual words")
#     histograms = create_image_histogram(visual_words, features)
#     features = None
#     print("created image histograms")
#     train_knn(histograms, 30, 500)
#     print("finished training")


def extract_Validations(histograms, validation_num):
    train_histograms = []
    validation_histograms = []

    for category in histograms:
        category_arr = np.array(category)
        validation_indexes = np.random.choice(len(category), validation_num)
        validation_category = category_arr[validation_indexes]
        train_category = category_arr[~validation_indexes]
        train_histograms.append(train_category.tolist())
        validation_histograms.append(validation_category.tolist())
    return train_histograms, validation_histograms


def ready_test_data(hyper_parameter, best_knn):
    test_path = "/home/sahel/university/vision/HW3/Data/Test/*/"
    features = extract_image_features(read_data(test_path))
    histograms = create_image_histogram(hyper_parameter.get("visual_words"), features)
    validation_xs = []
    validation_ys = []
    for i, validation_category in enumerate(histograms):
        validation_y = np.repeat(i, len(validation_category))
        validation_xs.extend(validation_category)
        validation_ys.extend(validation_y.tolist())
    score = best_knn.score(validation_xs, validation_ys)
    return score


def tune_homography():
    features = extract_image_features(read_data(INPUT_PATH))
    cluster_max = 150
    cluster_min = 70
    cluster_step = 10
    max_k = 40
    min_k = 5
    step = 5
    validation_num = 600
    best_score = -1
    best_model = None
    hyper_parameters = {}
    for cluster_num in range(cluster_min, cluster_max, cluster_step):
        visual_words = create_visual_words(features, cluster_num)
        histograms = create_image_histogram(visual_words, features)
        print("created histogram for cluster num : ", cluster_num)
        train_histograms, validation_histograms = extract_Validations(histograms, validation_num)
        for k in range(min_k, max_k, step):
            score, model = train_knn(train_histograms, k, validation_histograms)
            print("current score : ", score)
            if score > best_score:
                print("best score is : ", score)
                best_score = score
                best_model = model
                hyper_parameters = {"k": k, "cluster_num": cluster_num,
                                    "validation_num": validation_num}
                print(hyper_parameters)
        if hyper_parameters.get("cluster_num") == cluster_num:
            hyper_parameters["visual_words"] = visual_words
    print(hyper_parameters)
    return best_model, hyper_parameters


def q24():
    knn, hp = tune_homography()
    print(hp)
    print(ready_test_data(hp, knn))
    with open("hyper_parameters42.txt", "wt") as f:
        json.dump(hp, f)

"""
d = read_data(INPUT_PATH)
print("read data finished")
features = extract_image_features(d)
d = None
print("extracted features")
visual_words = create_visual_words(features, 50)
print("get visual words")
histograms = create_image_histogram(visual_words, features)
features = None
print("created image histograms")
train_knn(histograms, 30, 500)
print("finished training")
"""

# test_code()
# tune_homography()
q24()