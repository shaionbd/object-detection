import math
import os
import pickle

import face_recognition
from PIL import Image, ImageDraw
import time

from face_recognition import face_distance


class FaceRecognition:

    def __init__(self, known_data_directory="../images/known-images", img_size=128, stream=False):
        self.KNOWN_DATA_DIR = known_data_directory
        self.IMG_SIZE = img_size
        self.stream = stream

    def face_encoding_list(self, known_data_dir):
        known_face_encoding_list = []
        known_face_names = []

        for img in os.listdir(known_data_dir):
            known_image = face_recognition.load_image_file(os.path.join(known_data_dir, img))
            name_list = str(img).split('.')
            name = ''.join(name_list[:len(name_list)-1])
            known_face_names.append(name)
            known_face_encoding = face_recognition.face_encodings(known_image)[0]
            known_face_encoding_list.append(known_face_encoding)

        return known_face_encoding_list, known_face_names

    def face_encoding(self, image):
        try:
            img = face_recognition.load_image_file(image)
            return face_recognition.face_encodings(img)[0]
        except Exception as e:
            return False

    def save_data(self, data, names=None):
        # save in pickle file
        pickle_out = open("./data/knowns.pickle", "wb")
        pickle.dump(data, pickle_out)
        pickle_out.close()
        if names:
            pickle_out = open("./data/known_names.pickle", "wb")
            pickle.dump(names, pickle_out)
            pickle_out.close()

    def open_encoding_data(self):
        try:
            if self.stream:
                data, names = self.face_encoding_list(self.KNOWN_DATA_DIR)
                self.save_data(data, names)
                return data, names
            pickle_knowns_in = open("../data/knowns.pickle", "rb")
            pickle_known_names_in = open("../data/known_names.pickle", "rb")
            return pickle.load(pickle_knowns_in), pickle.load(pickle_known_names_in)
        except Exception as e:
            data, names = self.face_encoding_list(self.KNOWN_DATA_DIR)
            self.save_data(data, names)
            return data, names

    def get_compare_result(self, unknown):

        face_encoding_data, face_names = self.open_encoding_data()
        results = face_recognition.compare_faces(face_encoding_data, unknown)
        response = []
        for index in range(len(results)):
            response.append((face_names[index], results[index]))
        return response

    def face_distance_to_conf(self, face_distances, face_match_threshold=0.6):
        distances = []
        for fd in face_distances:
            if fd > face_match_threshold:
                range = (1.0 - face_match_threshold)
                linear_val = (1.0 - fd) / (range * 1.8)
                distances.append(linear_val)
            else:
                range = face_match_threshold
                linear_val = 1.0 - (fd / (range * 2.0))
                linear_val = linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2.0, 0.2))
                distances.append(linear_val)
        return distances

    def compare_faces(self, known_face_encodings, face_encoding_to_check, tolerance=0.6):
        """
        Compare a list of face encodings against a candidate encoding to see if they match.

        :param known_face_encodings: A list of known face encodings
        :param face_encoding_to_check: A single face encoding to compare against the list
        :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
        :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
        """
        face_accuracy = face_distance(known_face_encodings, face_encoding_to_check)

        return list(face_accuracy <= tolerance), self.face_distance_to_conf(face_accuracy, tolerance)

    def draw_face(self, draw, left, right, top, bottom, name, accuracy):
        # draw box in matches face
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 0))
        # draw label
        text_width_1, text_height_1 = draw.textsize(name)
        text_width_2, text_height_2 = draw.textsize(accuracy)
        draw.rectangle(((left, bottom + text_height_1 + 15), (right, bottom)), fill=(100, 100, 100), outline=(0, 0, 0))
        draw.text((left + 6, bottom + text_height_1 - 1), accuracy + '%', fill=(222, 222, 222))
        # name = str(name).capitalize() if len(name) <= 20 else str(name[0:20])+'...'
        draw.text((left + 6, bottom + text_height_2 - 10), name, fill=(222, 222, 222))

    def get_face_recognition(self, test_image, tolerance=0.55, min_accuracy=0.6, display=True):
        unknown_img = face_recognition.load_image_file(test_image)
        # Image encoding return face encoding array
        unknown_face_encoding = face_recognition.face_encodings(unknown_img)
        face_locations = face_recognition.face_locations(unknown_img)

        known_face_encoding_list, known_face_names = self.open_encoding_data()

        # (top, right, bottom, left), face_encoding = face_locations[0], unknown_face_encoding[0]
        #
        pil_img = Image.fromarray(unknown_img)
        draw = ImageDraw.Draw(pil_img)

        output_users = {}
        x = 0

        for (top, right, bottom, left), face_encoding in zip(face_locations, unknown_face_encoding):
            matches, accuracies = self.compare_faces(known_face_encoding_list, face_encoding, tolerance)
            top = top - int((bottom - top) / 4)
            right = right + int((right - left) / 4)
            bottom = bottom + int((bottom - top) / 4)
            left = left - int((right - left) / 4)

            x += 1
            # check if matches a face
            if True in matches and max(accuracies) > min_accuracy:
                max_index = accuracies.index(max(accuracies))
                name = known_face_names[max_index]
                accuracy = '%.2f' % (accuracies[max_index] * 100)
                if name not in output_users:
                    output_users[name] = {'accuracy': accuracy, 'left': left, 'right': right, 'top': top, 'bottom': bottom}
                else:
                    if float(output_users[name]['accuracy']) < float(accuracy):
                        temp = output_users[name]
                        temp['accuracy'] = 'XX'
                        output_users[name] = {'accuracy': accuracy, 'left': left, 'right': right, 'top': top, 'bottom': bottom}
                        output_users['Unknown Person {}'.format(x)] = temp
            else:
                output_users['Unknown Person {}'.format(x)] = {'accuracy': '%.2f' % (max(accuracies) * 100), 'left': left, 'right': right, 'top': top, 'bottom': bottom}
        if display:
            for face in output_users:
                self.draw_face(draw, output_users[face]['left'], output_users[face]['right'], output_users[face]['top'], output_users[face]['bottom'], face, output_users[face]['accuracy'])

        del draw
        # display image
        if display:
            pil_img.show()

        return output_users

    def nid_verification(self, upload_image, nid_no, display=False):
        result = self.get_face_recognition(upload_image, display=display)
        for face in result:
            face_nid = str(face).split('_')[0]
            if face_nid == nid_no:
                return {"match": True, "accuracy": result[face]['accuracy']}
        return {"match": False, "accuracy": '0'}


if __name__ == '__main__':
    try:
        start_time = time.time()
        fr = FaceRecognition()
        # l, n = fr.open_encoding_data()
        # print(n)
        # fr.face_crop(image='images/unknown-images/mukai_1.jpg', output_dir="images/known_images")
        #
        # data, names = fr.face_encoding_list(known_data_dir='./images/known_images')
        # fr.save_data(data, names)
        # test image
        unknown_image = '../images/unknown-images/bipul_khan_test2.jpg'

        # # unknown image file loading
        result = fr.nid_verification(unknown_image, 'bipul', display=True)
        print(result)
        end_time = time.time()
        print("Total Time Takes: {}".format(end_time - start_time))

    except Exception as e:
        print(e)
        print("Must input a valid image")



