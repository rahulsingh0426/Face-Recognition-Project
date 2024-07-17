import cv2
import logging
import face_recognition
from utils.capture import capture_frames
from utils.csv_utils import save_faces_to_csv, load_faces_from_csv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main(source=0):
    known_encodings, known_names = load_faces_from_csv()
    logging.info("Loaded known faces from CSV.")
    
    for frame in capture_frames(source):
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        if face_encodings:
            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = face_distances.argmin() if face_distances.size > 0 else None

                name = "Unknown"
                if best_match_index is not None and face_distances[best_match_index] < 0.6:
                    name = known_names[best_match_index]
                    logging.info(f"Recognized face: {name} with distance: {face_distances[best_match_index]}")
                    # Draw a green rectangle around recognized faces
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                else:
                    name = input("Enter the name for the detected unknown face: ")
                    logging.info(f"Unrecognized face detected and saved to CSV as {name}.")
                    save_faces_to_csv([face_encoding], [name])
                    # Draw a red rectangle around unrecognized faces
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                
                # Display the name below the rectangle
                cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Display the frame with rectangles
        cv2.imshow('Frame', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
