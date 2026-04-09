import os
import cv2
import csv
import mediapipe as mp
import numpy as np

# Configuration
DATASET_DIR = 'model/data'  # Cartella con le sottocartelle 0_Open, 1_Closed, ecc.
OUTPUT_CSV = './keypoint.csv'


class DatasetGenerator:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7
        )

    def get_frame_coordinates(self, landmark, width, height):
        """Converts normalized MediaPipe landmarks to pixel coordinates."""
        x = min(int(landmark.x * width), width - 1)
        y = min(int(landmark.y * height), height - 1)
        return np.array([x, y])

    def normalize_landmarks(self, landmark_list):
        """
        Normalizes landmarks relative to the wrist (landmark 0)
        and scales them by the maximum distance.
        """
        base_x, base_y = landmark_list[0][0], landmark_list[0][1]

        # Shift coordinates relative to wrist
        shifted_landmarks = []
        for point in landmark_list:
            shifted_landmarks.append([point[0] - base_x, point[1] - base_y])

        # Flatten to 1D list
        flattened = [val for sublist in shifted_landmarks for val in sublist]

        # Scale values between -1 and 1
        max_val = max(map(abs, flattened))
        if max_val == 0: return flattened  # Avoid division by zero

        return [v / max_val for v in flattened]

    def process_images(self):
        """Iterates through folders and processes each image."""
        if not os.path.exists(DATASET_DIR):
            print(f"Error: Directory {DATASET_DIR} not found.")
            return

        # Open CSV in write mode to start fresh
        with open(OUTPUT_CSV, 'w', newline="") as f:
            writer = csv.writer(f)

            for folder_name in sorted(os.listdir(DATASET_DIR)):
                folder_path = os.path.join(DATASET_DIR, folder_name)
                if not os.path.isdir(folder_path):
                    continue

                # Extract label index from folder name (e.g., "0" from "0_Open")
                try:
                    label = int(folder_name.split('_')[0])
                except (ValueError, IndexError):
                    print(f"Skipping folder with invalid name format: {folder_name}")
                    continue

                print(f"Processing label {label}: {folder_name}...")

                for img_name in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img_name)
                    image = cv2.imread(img_path)
                    if image is None: continue

                    h, w, _ = image.shape
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(rgb_image)

                    if results.multi_hand_landmarks:
                        for hand_lms in results.multi_hand_landmarks:
                            # 1. Get raw pixel coordinates
                            raw_coords = [self.get_frame_coordinates(lm, w, h) for lm in hand_lms.landmark]

                            # 2. Normalize
                            processed_data = self.normalize_landmarks(raw_coords)

                            # 3. Write to CSV (Label + 42 coordinates)
                            writer.writerow([label, *processed_data])

        print(f"Dataset successfully created: {OUTPUT_CSV}")


if __name__ == "__main__":
    generator = DatasetGenerator()
    generator.process_images()