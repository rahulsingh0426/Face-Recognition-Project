import cv2
import logging
import face_recognition
from utils.capture import capture_frames
from utils.csv_utils import save_faces_to_csv, load_faces_from_csv, log_recognized_face

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main(source=0):
    known_encodings, known_names = load_faces_from_csv()
    logging.info("Loaded known faces from CSV.")

    # Open the video source
    video_capture = cv2.VideoCapture(source)
    
    frame_count = 0
    frame_skip = 5  # Process every 5th frame

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        if face_encodings:
            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = face_distances.argmin() if face_distances.size > 0 else None

                name = "Unknown"
                if best_match_index is not None and face_distances[best_match_index] < 0.6:
                    name = known_names[best_match_index]
                    logging.info(f"Recognized {name} with distance: {face_distances[best_match_index]}")
                    log_recognized_face(name)
                    # Draw a green rectangle around recognized faces
                    color = (0, 255, 0)
                else:
                    name = input("Enter name for the unrecognized face: ")
                    known_encodings.append(face_encoding)
                    known_names.append(name)
                    save_faces_to_csv([(name, face_encoding)])
                    logging.info(f"Unrecognized face saved as {name}.")
                    # Draw a red rectangle around unrecognized faces
                    color = (0, 0, 255)
                
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Draw rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                # Display the name below the rectangle
                cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Display the frame with rectangles
        cv2.imshow('Frame', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()