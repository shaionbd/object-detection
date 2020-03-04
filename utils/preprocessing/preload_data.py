import os
import re
import urllib.request
import pickle
import numpy as np


class PreloadData:

    def __init__(self, source=None, destination=None, filename=None):
        self.source = source
        self.destination = destination
        self.filename = filename
        self.data = None

    def is_link(self):
        regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return re.match(regex, self.source) is not None

    def rescale_data(self, data,  rescale):
        return data * float(rescale)

    def load_path(self):
        # url = "https://github.com/shaionbd/face-recognition/raw/master/data/knowns.pickle"
        # data = load_data_from_url(url=link)
        try:
            if not self.filename:
                self.filename = str(self.source).split("/")[-1]
            if self.destination:
                self.filename = os.path.join(self.destination, self.filename)
            urllib.request.urlretrieve(self.source, self.filename)
            self.source = self.filename
        except Exception as e:
            print(e)

    def label_indexing(self, labels):
        indexes = np.unique(labels, return_inverse=True)
        # return class name, class value
        return indexes[0], np.asarray(indexes[1], dtype=np.float32)

    def label_processing(self, labels, remove_part="", remove_regex=''):
        # remove_part="PetImages", remove_regex='[0-9]+'
        new_labels = []
        for label in labels:
            if type(remove_part) == list:
                for rm in remove_part:
                    label = label.replace(rm, "")
            else:
                label = label.replace(remove_part, "").replace("_", "")
            label = re.sub(remove_regex, '', label)
            new_labels.append(label)
        return new_labels

    def combine_dataset(self, features, labels):
        return [[feature, label] for feature, label in zip(features, labels)]

    def seperate_dataset(self, datasets) -> tuple:
        features = []
        labels = []
        for feature, label in datasets:
            features.append(feature)
            labels.append(label)
        return features, labels

    def shuffle(self, features, labels, random_state=None, repeat=1):
        dataset = self.combine_dataset(features, labels)
        if random_state:
            np.random.seed(random_state)
        for _ in range(repeat):
            np.random.shuffle(dataset)
        return self.seperate_dataset(dataset)

    def shuffle_dataset(self, dataset, random_state=None, repeat=1):
        if random_state:
            np.random.seed(random_state)
        for _ in range(repeat):
            np.random.shuffle(dataset)
        return dataset

    def train_test_split(self, features, labels, test_size=0.33, random_state=None):
        features, labels = self.shuffle(features, labels, random_state)
        data_len = len(labels)
        test_len = int(data_len * test_size)
        train_data, train_label, test_data, test_label = features[:(data_len - test_len)], labels[:(
                    data_len - test_len)], features[-test_len:], labels[-test_len:]

        return np.asarray(train_data, dtype=np.float32), np.asarray(train_label, dtype=np.float32), np.asarray(
            test_data, dtype=np.float32), np.asarray(test_label, dtype=np.float32)

    def validate_data(self, features, labels, validate_size=.2):
        data_len = len(labels)
        validate_len = int(data_len * validate_size)
        train_data, train_label, validate_data, validate_label = features[:(data_len - validate_len)], labels[:(
                    data_len - validate_len)], features[-validate_len:], labels[-validate_len:]

        return np.asarray(train_data, dtype=np.float32), np.asarray(train_label, dtype=np.float32), np.asarray(
            validate_data, dtype=np.float32), np.asarray(validate_label, dtype=np.float32)

    def load_data(self):
        if self.is_link():
            self.load_path()
        if os.path.isfile(self.source):
            try:
                pickle_data_in = open(self.source, "rb")
                self.data = pickle.load(pickle_data_in)
                return self.data
            except Exception as e:
                return False
        else:
            print("Extract Data Failed")
            return False

    def save(self, data=None, destination=None, filename=None):
        if not destination:
            destination = self.destination if self.destination else '.'
        if not filename:
            filename = self.filename if self.filename else 'data.pickle'

        if not os.path.isdir(destination) and destination != '.':
            os.makedirs(destination, exist_ok=True)
        output = os.path.join(destination, filename)
        pickle_out = open(output, "wb")

        if data is None:
            data = self.data
        pickle.dump(data, pickle_out)
        pickle_out.close()

