import os
import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from utils.preprocessing.preload_data import PreloadData


class ImagePreprocessing(PreloadData):

    def __init__(self, source, destination=None, filename=None, IMG_SIZE=50, resize_image=True, original_image=False,
                 rescale=None):
        super().__init__(source, destination, filename)
        self.source = source
        self.destination = destination
        self.filename = filename
        self.data = None
        self.IMG_SIZE = IMG_SIZE
        self.resize_image = resize_image
        self.original_image = original_image
        self.rescale = rescale

    def __save_face_crop(self, position: tuple, known_image, image_name, output_dir):
        top, right, bottom, left = position
        top = top - int((bottom - top) / 4)
        right = right + int((right - left) / 4)
        bottom = bottom + int((bottom - top) / 4)
        left = left - int((right - left) / 4)
        pil_img = Image.fromarray(known_image)
        crop_img = pil_img.crop((left, top, right, bottom))
        gray_img = ImageOps.grayscale(crop_img)
        if output_dir:
            gray_img.save(output_dir + '/' + image_name)
        else:
            gray_img.save(known_image)

    def face_crop(self, known_data_dir=None, image=None, output_dir=None):
        if known_data_dir or image is None:
            known_data_dir = known_data_dir if known_data_dir else self.source
            for img in tqdm(os.listdir(known_data_dir)):
                known_image = face_recognition.load_image_file(os.path.join(known_data_dir, img))
                face_location = face_recognition.face_locations(known_image)
                if not face_location:
                    print("Image quality of {} is not good. Please pass a valid image".format(img))
                else:
                    self.__save_face_crop(face_location[0], known_image, img, output_dir)
            print("Image has cropped")
        else:
            known_image = face_recognition.load_image_file(image)
            img_path_list = str(image).split("/")
            img_name = img_path_list[len(img_path_list) - 1]
            face_location = face_recognition.face_locations(known_image)
            if not face_location:
                print("Image quality is not good. Please pass a valid image")
            else:
                self.__save_face_crop(face_location[0], known_image, img_name, output_dir)
                print("Image has cropped")

    def get_image_array(self, source):
        try:
            if self.original_image:
                img_array = cv2.imread(source)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            else:
                img_array = cv2.imread(source, cv2.IMREAD_GRAYSCALE)  # convert to array

            if self.resize_image:
                img_array = cv2.resize(img_array, (self.IMG_SIZE, self.IMG_SIZE))
            return img_array
        except Exception as e:
            # print("Corrupted Image: {}".format(source))
            return False

    def __create_data(self, source, image_data=None):
        if image_data is None:
            image_data = []
        for img in tqdm(os.listdir(source)):
            if os.path.isdir(os.path.join(source, img)):
                self.__create_data(os.path.join(source, img), image_data)
            else:
                img_src = str(img).split('.')
                img_name = ''.join(img_src[: len(img_src) - 1])
                src_name = str(source).replace('/', '_')
                src_name = str(src_name).replace('._', '')
                img_label = src_name + '_' + img_name
                img_array = self.get_image_array(os.path.join(source, img))
                if img_array is False:
                    pass
                else:
                    image_data.append([img_array, img_label])
        return image_data

    def extract_model(self, training_data, reshape=False, reshape_size=None) -> tuple:
        X = []
        y = []
        for features, labels in training_data:
            X.append(features)
            y.append(labels)
        if reshape:
            if reshape_size is None:
                reshape_size = self.IMG_SIZE
            X = np.array(X).reshape(-1, reshape_size, reshape_size, 1)

        return X, y

    def reshape(self, features=None, reshape_size=None):
        if reshape_size is None:
            reshape_size = self.IMG_SIZE
        if features is None:
            features = self.data[0]
        return np.array(features).reshape(-1, reshape_size, reshape_size, 1)

    def extract_data(self, reshape=False, reshape_size=None, shuffle=False, random_state=None, repeat=1):
        if os.path.isdir(self.source):
            training_data = self.__create_data(self.source)
            if shuffle:
                training_data = self.shuffle_dataset(dataset=training_data, random_state=random_state, repeat=repeat)
            self.data = self.extract_model(training_data=training_data, reshape=reshape, reshape_size=reshape_size)
        else:
            features, labels = self.load_data()
            if shuffle:
                features, labels = self.shuffle(features=features, labels=labels, random_state=random_state, repeat=repeat)
            if reshape:
                features = self.reshape(features=features, reshape_size=reshape_size)
            self.data = (features, labels)
        return self.data

    def test_image_preparation(self, img_path):
        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (self.IMG_SIZE, self.IMG_SIZE))
        return new_array.reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
