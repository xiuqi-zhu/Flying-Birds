import os
import argparse
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torch
from facenet_pytorch import MTCNN as TorchMTCNN


def setup_detectors():
    """Initialize Haar Cascade and MTCNN detectors."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise ValueError("Failed to load Haar Cascade classifier.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn_detector = TorchMTCNN(image_size=224, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], device=device)
    return face_cascade, mtcnn_detector


def extract_and_fill_frames(video_path, num_frames=12):
    """
    Extract frames with faces from a video and fill missing frames.

    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to extract (default: 12).

    Returns:
        list: List of extracted frames (or None for invalid frames).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames)
    frame_positions = [i * frame_interval for i in range(num_frames)]
    frame_results = []

    # Process each frame position
    for pos in frame_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret or frame is None:
            frame_results.append(None)
            continue

        # Dual detection: Haar Cascade and MTCNN
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        haar_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        boxes, _ = mtcnn_detector.detect(pil_img)

        if len(haar_faces) > 0 and boxes is not None and len(boxes) > 0:
            frame_results.append(frame)
            print(f"Frame at position {pos}: Detected faces by both methods")
        else:
            frame_results.append(None)
            print(f"Frame at position {pos}: No faces detected by both methods")

    cap.release()

    # Return empty list if no valid frames
    if not any(frame is not None for frame in frame_results):
        print(f"No valid faces detected in video: {video_path}")
        return []

    # Fill invalid frames with previous or next valid frame
    filled_frames = []
    last_valid_frame = None

    for i, frame in enumerate(frame_results):
        if frame is not None:
            filled_frames.append(frame)
            last_valid_frame = frame
        else:
            if last_valid_frame is not None:
                filled_frames.append(last_valid_frame)
                print(f"Frame {i + 1}: Copied previous valid frame")
            else:
                next_valid_frame = next((f for f in frame_results[i + 1:] if f is not None), None)
                filled_frames.append(next_valid_frame if next_valid_frame is not None else last_valid_frame)
                print(f"Frame {i + 1}: Copied {'next' if next_valid_frame else 'last'} valid frame")

    # Ensure all frames are filled with the last valid frame if needed
    filled_frames = [frame if frame is not None else last_valid_frame for frame in filled_frames]
    return filled_frames[:num_frames]


def extract_face_from_frame(frame, face_cascade, mtcnn_detector, img_size=(224, 224)):
    """
    Extract the largest face from a frame using dual detection.

    Args:
        frame (numpy.ndarray): Input frame.
        face_cascade: Haar Cascade classifier.
        mtcnn_detector: MTCNN detector.
        img_size (tuple): Target size for resized face (default: (224, 224)).

    Returns:
        numpy.ndarray: Resized face image or None if no face detected.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    haar_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    boxes, _ = mtcnn_detector.detect(pil_img)

    if len(haar_faces) == 0 or boxes is None or len(boxes) == 0:
        return None

    # Select the largest face
    mtcnn_areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in boxes]
    max_idx = np.argmax(mtcnn_areas)
    x1, y1, x2, y2 = boxes[max_idx]
    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

    # Calculate square region with padding
    side = max(w, h)
    padding = int(side * 0.2)
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(frame.shape[1], x + side + padding)
    y_end = min(frame.shape[0], y + side + padding)

    face_img = frame[y_start:y_end, x_start:x_end]
    if face_img.size == 0:
        return None

    # Resize to square
    if face_img.shape[0] != face_img.shape[1]:
        max_side = max(face_img.shape[0], face_img.shape[1])
        padded_img = np.zeros((max_side, max_side, 3), dtype=np.uint8)
        offset_y = (max_side - face_img.shape[0]) // 2
        offset_x = (max_side - face_img.shape[1]) // 2
        padded_img[offset_y:offset_y + face_img.shape[0], offset_x:offset_x + face_img.shape[1]] = face_img
        face_img = padded_img

    face_resized = cv2.resize(face_img, img_size)
    return face_resized


def save_faces_as_images(faces, output_folder, folder_name, ab_folder_name, num_frames=12):
    """
    Save extracted faces as images.

    Args:
        faces (list): List of face images.
        output_folder (str): Base output directory.
        folder_name (str): Folder for video ID.
        ab_folder_name (str): Subfolder for clip ID.
        num_frames (int): Number of frames to save (default: 12).

    Returns:
        int: Number of successfully saved images.
    """
    folder_path = os.path.join(output_folder, folder_name, ab_folder_name)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Created/Using folder: {folder_path}")

    saved_count = 0
    for idx, face in enumerate(faces[:num_frames]):
        if face is not None:
            filename = os.path.join(folder_path, f"{ab_folder_name}_face_{idx + 1}.jpg")
            if cv2.imwrite(filename, face):
                saved_count += 1
                print(f"Saved {filename}")
            else:
                print(f"Failed to save {filename}")
        else:
            print(f"Frame {idx + 1}: No valid face to save")

    return saved_count


def process_video(video_id, clip_id, video_base_path, output_folder, face_cascade, mtcnn_detector, num_frames=12,
                  img_size=(224, 224)):
    """
    Process a single video to extract and save faces.

    Args:
        video_id (str): Video ID.
        clip_id (str): Clip ID.
        video_base_path (str): Base path for video files.
        output_folder (str): Base path for output.
        face_cascade: Haar Cascade classifier.
        mtcnn_detector: MTCNN detector.
        num_frames (int): Number of frames to process (default: 12).
        img_size (tuple): Target size for face images (default: (224, 224)).
    """
    video_name = f"{video_id}_{clip_id}.mp4"
    video_path = os.path.join(video_base_path, video_name)
    folder_name = str(video_id)
    ab_folder_name = f"{video_id}_{clip_id}"

    if not os.path.exists(video_path):
        print(f"Video {video_name} not found at: {video_path}")
        return

    print(f"\nProcessing video: {video_name}")
    frames = extract_and_fill_frames(video_path, num_frames)

    if not frames:
        print(f"No frames processed for video: {video_name}")
        return

    # Extract faces
    faces = []
    for idx, frame in enumerate(frames):
        face = extract_face_from_frame(frame, face_cascade, mtcnn_detector, img_size)
        faces.append(face)
        print(f"Frame {idx + 1}: {'Face extracted' if face is not None else 'No face detected'}")

    # Save face images
    saved_count = save_faces_as_images(faces, output_folder, folder_name, ab_folder_name, num_frames)
    print(f"Completed processing {video_name}: Saved {saved_count}/{num_frames} faces")


def main():
    """Main function to process videos based on CSV input."""
    parser = argparse.ArgumentParser(description="Extract faces from video clips using Haar Cascade and MTCNN.")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to the CSV file with video metadata.")
    parser.add_argument('--video_base_path', type=str, required=True, help="Base directory for video files.")
    parser.add_argument('--output_folder', type=str, default="frames_output",
                        help="Output directory for extracted faces.")
    args = parser.parse_args()

    # Set environment variable to avoid OpenMP conflict
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Load CSV
    try:
        df = pd.read_csv(args.csv_path)
        print("Loaded CSV with video metadata:")
        print(df[['video_id', 'clip_id']])
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Initialize detectors
    try:
        face_cascade, mtcnn_detector = setup_detectors()
    except Exception as e:
        print(f"Error initializing detectors: {e}")
        return

    # Process all videos
    total_clips = len(df)
    print(f"Found {total_clips} video clips to process")
    for index, row in df.iterrows():
        video_id = row['video_id']
        clip_id = row['clip_id']
        print(f"Processing clip {index + 1}/{total_clips}: {video_id}_{clip_id}")
        process_video(video_id, clip_id, args.video_base_path, args.output_folder, face_cascade, mtcnn_detector)


if __name__ == '__main__':
    main()