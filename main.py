import tensorflow as tf
import numpy as np
from PIL import Image
import os

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_path, size):
    image = Image.open(image_path)
    image = image.resize(size)
    image = np.array(image)
    image = (image.astype(np.float32) - 127.5) / 127.5
    image = np.expand_dims(image, axis=0)
    return image

def generate_ai_art(model, image_path, output_path):
    model = load_model(model_path)

    image_size = (256, 256)
    input_image = preprocess_image(image_path, image_size)

    generated_image = model.predict(input_image)

    generated_image = ((generated_image[0] * 0.5 + 0.5) * 255).astype(np.uint8)

    generated_image = Image.fromarray(generated_image)

    generated_image.save(output_path)
    print("AI art generated and saved successfully!")


def resize_images(input_dir, output_dir, size):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        img = Image.open(image_path)
        img = img.resize(size)
        img.save(output_path)


def normalize_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        img = Image.open(image_path)
        img = np.array(img)
        img = (img.astype(np.float32) - 127.5) / 127.5  
        img = (img.astype(np.float32) - img.min()) / (img.max() - img.min())  
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(output_path)


model_path = "path/to/pretrained/model.h5"
input_image_path = "path/to/input/image.jpg"
output_image_path = "path/to/output/art.png"
input_dir = "path/to/original/images/"
output_dir = "path/to/resized/images/"
normalized_dir = "path/to/normalized/images/"
image_size = (256, 256)

resize_images(input_dir, output_dir, image_size)

normalize_images(output_dir, normalized_dir)

generate_ai_art(model_path, input_image_path, output_image_path)