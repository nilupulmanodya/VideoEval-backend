# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import cv2
import mediapipe as mp
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
import os
import tempfile
from supabase import create_client, Client
from dotenv import load_dotenv


app = Flask(__name__)
CORS(app)  # Add this line right after creating the Flask app


load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

print("supabase", SUPABASE_URL)

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variables")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
BUCKET_NAME = "results"  # Replace with your bucket name

def evaluate_presentation(video_path):
    """
    Evaluates presentation skills from a video, tracking eye contact, hand movements, and audio,
    and returns a pandas DataFrame containing scores and timestamps.

    Args:
        video_path (str): Path to the video file.

    Returns:
        pandas.DataFrame: A DataFrame containing scores and timestamps.
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    eye_contact_scores = []
    hand_movement_scores = []
    audio_scores = []
    timestamps = []
    audio_data = []
    sr = None  # Initialize sr

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_face = face_mesh.process(frame_rgb)
            results_hands = hands.process(frame_rgb)

            # Eye Contact Evaluation
            eye_contact_score = 0
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    left_eye_visible = True  # Placeholder, improve with better eye tracking
                    right_eye_visible = True # placeholder, improve with better eye tracking
                    if left_eye_visible and right_eye_visible:
                        eye_contact_score = 1
                    else:
                        eye_contact_score = 0
            eye_contact_scores.append(eye_contact_score)

            # Hand Movement Evaluation
            hand_movement_score = 0
            if results_hands.multi_hand_landmarks:
                hand_movement_score = 1
            hand_movement_scores.append(hand_movement_score)

            # Audio Extraction
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            time_seconds = frame_number / fps
            try:
                audio_clip, current_sr = librosa.load(video_path, sr=None, offset=time_seconds, duration=1/fps)
                audio_data.extend(audio_clip.tolist())
                if sr is None and current_sr is not None:
                    sr = current_sr
            except Exception as e:
                print(f"Error extracting audio at {time_seconds}: {e}")
                # Pad with zeros based on a default or previously obtained sr
                if sr is not None and fps > 0:
                    audio_data.extend([0.0] * int(sr / fps))
                elif fps > 0:
                    audio_data.extend([0.0] * 441) # Default padding if sr is still None and fps > 0
                else:
                    audio_data.append(0.0) # Fallback if fps is also not available

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Audio Evaluation
    audio_data = np.array(audio_data)
    frame_per_sec = int(fps) if fps > 0 else 30 # Default fps if unavailable
    if sr is not None and frame_per_sec > 0:
        for i in range(0, len(audio_data), sr // frame_per_sec):
            audio_frame = audio_data[i: i + sr // frame_per_sec]
            if len(audio_frame) > 0:
                rms = np.sqrt(np.mean(audio_frame**2))
                if rms > 0.01: # adjust threshold as needed
                    audio_scores.append(1)
                else:
                    audio_scores.append(0)
            else:
                audio_scores.append(0)
    else:
        print("Warning: Audio sampling rate (sr) or frames per second (fps) not available for audio scoring.")
        audio_scores = [0] * int(duration) # Assign default audio scores

    # Align scores to seconds and create DataFrame
    eye_contact_scores_per_second = [np.mean(eye_contact_scores[int(i*fps):int((i+1)*fps)]) if fps > 0 and int((i+1)*fps) <= len(eye_contact_scores) else 0 for i in range(int(duration))]
    hand_movement_scores_per_second = [np.mean(hand_movement_scores[int(i*fps):int((i+1)*fps)]) if fps > 0 and int((i+1)*fps) <= len(hand_movement_scores) else 0 for i in range(int(duration))]
    audio_scores_per_second = audio_scores[:int(duration)]

    for i in range(int(duration)):
        timestamps.append(i)

    df = pd.DataFrame({
        'Timestamp (seconds)': timestamps,
        'Eye Contact Score': eye_contact_scores_per_second,
        'Hand Movement Score': hand_movement_scores_per_second,
        'Audio Score': audio_scores_per_second,
    })

    return df
    """
    Evaluates presentation skills from a video, tracking eye contact, hand movements, and audio,
    and returns a pandas DataFrame containing scores and timestamps.

    Args:
        video_path (str): Path to the video file.

    Returns:
        pandas.DataFrame: A DataFrame containing scores and timestamps.
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    eye_contact_scores = []
    hand_movement_scores = []
    audio_scores = []
    timestamps = []
    audio_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_face = face_mesh.process(frame_rgb)
        results_hands = hands.process(frame_rgb)

        # Eye Contact Evaluation
        eye_contact_score = 0
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                left_eye_visible = True  # Placeholder, improve with better eye tracking
                right_eye_visible = True # placeholder, improve with better eye tracking
                if left_eye_visible and right_eye_visible:
                    eye_contact_score = 1
                else:
                    eye_contact_score = 0
        eye_contact_scores.append(eye_contact_score)

        # Hand Movement Evaluation
        hand_movement_score = 0
        if results_hands.multi_hand_landmarks:
            hand_movement_score = 1
        hand_movement_scores.append(hand_movement_score)

        # Audio Extraction
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        time_seconds = frame_number / fps
        try:
            audio_clip, sr = librosa.load(video_path, sr=None, offset=time_seconds, duration=1/fps)
            audio_data.extend(audio_clip.tolist())
        except Exception as e:
            print(f"Error extracting audio at {time_seconds}: {e}")
            audio_data.extend([0.0] * int(sr / fps) if fps > 0 else []) # Pad with zeros if error

    cap.release()
    cv2.destroyAllWindows()

    # Audio Evaluation
    audio_data = np.array(audio_data)
    frame_per_sec = int(fps) if fps > 0 else 30 # Default fps if unavailable
    for i in range(0, len(audio_data), sr // frame_per_sec if sr > 0 and frame_per_sec > 0 else 1):
        audio_frame = audio_data[i: i + sr // frame_per_sec if sr > 0 and frame_per_sec > 0 else []]
        if len(audio_frame) > 0:
            rms = np.sqrt(np.mean(audio_frame**2))
            if rms > 0.01: # adjust threshold as needed
                audio_scores.append(1)
            else:
                audio_scores.append(0)
        else:
            audio_scores.append(0)

    # Align scores to seconds and create DataFrame
    eye_contact_scores_per_second = [np.mean(eye_contact_scores[int(i*fps):int((i+1)*fps)]) if fps > 0 and int((i+1)*fps) <= len(eye_contact_scores) else 0 for i in range(int(duration))]
    hand_movement_scores_per_second = [np.mean(hand_movement_scores[int(i*fps):int((i+1)*fps)]) if fps > 0 and int((i+1)*fps) <= len(hand_movement_scores) else 0 for i in range(int(duration))]
    audio_scores_per_second = audio_scores[:int(duration)]

    for i in range(int(duration)):
        timestamps.append(i)

    df = pd.DataFrame({
        'Timestamp (seconds)': timestamps,
        'Eye Contact Score': eye_contact_scores_per_second,
        'Hand Movement Score': hand_movement_scores_per_second,
        'Audio Score': audio_scores_per_second,
    })

    return df

def download_video(video_url):
    """Downloads a video from a given URL to a temporary file."""
    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        with open(temp_file.name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return temp_file.name
    except requests.exceptions.RequestException as e:
        print(f"Error downloading video from {video_url}: {e}")
        return None

def upload_to_supabase(file_path, bucket_name, object_name):
    """Uploads a file to Supabase storage."""
    try:
        with open(file_path, 'rb') as f:
            file_content = f.read()
        res = supabase.storage.from_(bucket_name).upload(object_name, file_content)
        print(f"Supabase Upload Response: {res}")  # Print the entire response for inspection
        if hasattr(res, 'error') and res.error:
            print(f"Error uploading to Supabase: {res.error}")
            return None
        else:
            upload_path = res.path  # Use the 'path' attribute

            return f"https://{SUPABASE_URL.split('//')[1]}/storage/v1/object/public/{bucket_name}/{upload_path}"
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred during Supabase upload: {e}")
        return None

@app.route('/evaluate', methods=['POST'])
def evaluate_video_webhook():
    """
    Webhook endpoint that receives a video URL, downloads the video,
    evaluates the presentation, uploads the results to Supabase, and
    returns the Supabase URL of the uploaded CSV.
    """
    data = request.get_json()
    print("Received data:", data)  # Debugging line to check incoming data
    if not data or 'video_url' not in data or 'video_id' not in data:
        return jsonify({"error": "Missing 'video_url' or 'video_id' in the request body"}), 400

    video_url = data['video_url']
    video_id = data['video_id']
    print(f"Received video URL: {video_url}, Video ID: {video_id}")

    # Download the video
    local_video_path = download_video(video_url)
    if not local_video_path:
        return jsonify({"error": "Failed to download the video"}), 500

    try:
        # Evaluate the presentation
        result_df = evaluate_presentation(local_video_path)

        # Save the DataFrame to a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".csv") as tmp_csv_file:
            csv_file_path = tmp_csv_file.name
            result_df.to_csv(csv_file_path, index=False)

        # Upload the CSV to Supabase
        csv_filename = f"presentation_scores_{os.path.basename(local_video_path).split('.')[0]}.csv"
        supabase_url = upload_to_supabase(csv_file_path, BUCKET_NAME, csv_filename)

        # Clean up temporary files
        os.remove(local_video_path)
        os.remove(csv_file_path)

        if supabase_url:
            # Update the videos table with the results URL
            try:
                update_response = supabase.table('videos').update({'results_url': supabase_url, 'status': "completed"}).eq('video_url', video_url).execute()
                if update_response.get('error'):
                    print(f"Error updating videos table: {update_response['error']}")
                    return jsonify({"error": "Failed to update the videos table with results URL"}), 500
            except Exception as e:
                print(f"An error occurred while updating the videos table: {e}")
                return jsonify({"error": "Failed to update the videos table with results URL"}), 500

            return jsonify({"message": "Evaluation successful, results uploaded to Supabase", "supabase_url": supabase_url}), 200
        else:
            return jsonify({"error": "Evaluation successful, but failed to upload results to Supabase"}), 500

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        if os.path.exists(local_video_path):
            os.remove(local_video_path)
        return jsonify({"error": f"Evaluation failed: {str(e)}"}), 500
    # Download the video
    local_video_path = download_video(video_url)
    if not local_video_path:
        return jsonify({"error": "Failed to download the video"}), 500

    try:
        # Evaluate the presentation
        result_df = evaluate_presentation(local_video_path)

        # Save the DataFrame to a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".csv") as tmp_csv_file:
            csv_file_path = tmp_csv_file.name
            result_df.to_csv(csv_file_path, index=False)

        # Upload the CSV to Supabase
        csv_filename = f"presentation_scores_{os.path.basename(local_video_path).split('.')[0]}.csv"
        supabase_url = upload_to_supabase(csv_file_path, BUCKET_NAME, csv_filename)

        # Clean up temporary files
        os.remove(local_video_path)
        os.remove(csv_file_path)

        if supabase_url:
            return jsonify({"message": "Evaluation successful, results uploaded to Supabase", "supabase_url": supabase_url}), 200
        else:
            return jsonify({"error": "Evaluation successful, but failed to upload results to Supabase"}), 500

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        if os.path.exists(local_video_path):
            os.remove(local_video_path)
        return jsonify({"error": f"Evaluation failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
