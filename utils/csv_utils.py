import csv
import datetime

def save_faces_to_csv(encodings_with_names, file_name="data/faces.csv"):
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        for name, encoding in encodings_with_names:
            row = [name, datetime.datetime.now()] + encoding.tolist()
            writer.writerow(row)

def load_faces_from_csv(file_name="data/faces.csv"):
    known_encodings = []
    known_names = []
    with open(file_name, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            name = row[0]
            encoding = [float(x) for x in row[2:]]
            known_names.append(name)
            known_encodings.append(encoding)
    return known_encodings, known_names

def log_recognized_face(name, file_name="data/recognized_faces.csv"):
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        row = [name, datetime.datetime.now()]
        writer.writerow(row)