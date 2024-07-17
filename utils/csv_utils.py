import csv
import datetime

def save_faces_to_csv(encodings, file_name="data/faces.csv"):
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        for encoding in encodings:
            row = [datetime.datetime.now()] + encoding.tolist()
            writer.writerow(row)

def load_faces_from_csv(file_name="data/faces.csv"):
    known_encodings = []
    known_times = []
    with open(file_name, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            time = row[0]
            encoding = [float(x) for x in row[1:]]
            known_times.append(time)
            known_encodings.append(encoding)
    return known_encodings, known_times
