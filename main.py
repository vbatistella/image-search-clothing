from tensorflow.keras.applications.resnet50 import ResNet50
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import json
import random
from scipy import spatial

IMG_SHAPE = (80, 60, 3)
CALCULATE_VECTORS = False

def show_image(data):
    plt.imshow(data)
    plt.show()

def get_files_from_dir(dir):
    return os.listdir(dir)

def get_image(file):
    return Image.open(file)

def get_image_data(file):
    return np.asarray(get_image(file))

def print_dataset_info(dir):
    a = get_files_from_dir("images")
    data = get_image_data(f"images/{a[0]}")
    print(f"total images   : {len(a)}")
    print(f"image height   : {len(data)}")
    print(f"image width    : {len(data[0])}")
    print(f"image channels : {len(data[0][0])}")

def get_vector(path, filename, model):
    data = get_image_data(path+"/"+filename)
    data = data.reshape((1, IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]))
    vector = model.predict(data)
    return vector[0]

def vector_all_images(dir, model):
    files = get_files_from_dir(dir)
    vectorized = []
    for f in files:
        print(f)
        try:
            v = get_vector(dir, f, model)
        except:
            continue
        vectorized.append({"file":f, "vector":v.tolist()})
    return vectorized

def save_vectors(vector, name):
    with open(name, 'w') as file:
        file.write(json.dumps({"images": vector}))

def distance(a, b):
    return spatial.distance.cosine(a, b)

def pick_random_from_set(path):
    files = get_files_from_dir(path)
    i = random.randint(0, len(files))
    return files[i]

def get_json(path):
    f = open('vectorized.json')
    return json.load(f)

def get_all_distances_from_vector(vector, all_vectors):
    for i in range(len(all_vectors)):
        d = distance(vector, all_vectors[i]["vector"])
        all_vectors[i]["vector"] = d
    return all_vectors

def order_by_distance(all_distances: list, key_value="vector"):
    l = all_distances
    l.sort(key=lambda d: d[key_value])
    return l

def plot_results(ref, similar):
    n = len(similar)
    fig = plt.figure(figsize=(10, 7))

    fig.add_subplot(2, n, (n//2)+1)
    plt.imshow(ref)
    plt.axis('off')
    plt.title("Reference")

    for i in range(n):
        fig.add_subplot(2, n, n+i+1)
        plt.imshow(similar[i])
        plt.axis('off')

    plt.show()

def main():
    print_dataset_info("images")
    model = ResNet50(include_top=False, input_shape=IMG_SHAPE, pooling='max')

    if CALCULATE_VECTORS:
        vect = vector_all_images("images", model)
        save_vectors(vect, "vectorized.json")

    filename = pick_random_from_set("images")
    # 5273.jpg / 7909.jpg / 25650.jpg
    # filename = "25650.jpg"
    im = get_image_data(f"images/{filename}")
    show_image(im)
    vector = get_vector("images", filename, model)
    all_vectors = get_json("vectorized.json")["images"]
    all_distances = get_all_distances_from_vector(vector, all_vectors)
    ordered = order_by_distance(all_distances, key_value="vector")
    top_5 = ordered[1:6]

    similar = []
    for i in top_5:
        file = i["file"]
        im = get_image_data(f"images/{file}")
        similar.append(im)
    
    plot_results(im, similar)

if __name__ == '__main__':
    main()