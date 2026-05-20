import face_recognition
import os
import pickle

DATASET_PATH = "dataset"
ENCODING_FILE = "encodings.pkl"


def generate_encodings():

    known_encodings = []
    known_names = []

    for file in os.listdir(DATASET_PATH):

        if file.endswith(".jpg") or file.endswith(".png"):

            path = os.path.join(DATASET_PATH, file)

            image = face_recognition.load_image_file(path)

            encoding = face_recognition.face_encodings(image)

            if encoding:
                known_encodings.append(encoding[0])
                name = file.split(".")[0]
                known_names.append(name)

    data = {
        "encodings": known_encodings,
        "names": known_names
    }

    with open(ENCODING_FILE, "wb") as f:
        pickle.dump(data, f)

    print("Encodings generated successfully.")


def load_encodings():

    with open(ENCODING_FILE, "rb") as f:
        data = pickle.load(f)

    return data