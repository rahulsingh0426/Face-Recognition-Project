import face_recognition

def detect_and_encode_faces(frame):
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    return face_locations, face_encodings

def recognize_faces_in_frame(frame, known_encodings):
    face_locations, face_encodings = detect_and_encode_faces(frame)
    matches = []
    for encoding in face_encodings:
        match = face_recognition.compare_faces(known_encodings, encoding)
        matches.append(match)
    return matches, face_locations
