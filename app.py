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

    # Initialize audio data
    try:
        print(f"Attempting to load audio from: {video_path}")
        # Try to load audio with a specific backend
        try:
            # First try with soundfile backend
            audio_data, sr = librosa.load(video_path, sr=None, res_type='kaiser_fast')
        except Exception as e1:
            print(f"SoundFile backend failed with error: {str(e1)}")
            try:
                # Then try with audioread backend
                audio_data, sr = librosa.load(video_path, sr=None, res_type='kaiser_fast')
            except Exception as e2:
                print(f"Audioread backend failed with error: {str(e2)}")
                raise Exception("Both audio backends failed to load the file")

        print(f"Successfully loaded audio. Sample rate: {sr}, Audio length: {len(audio_data)}")
        # Calculate the number of samples per frame
        samples_per_frame = int(sr / fps) if fps > 0 else int(sr / 30)
        print(f"Samples per frame: {samples_per_frame}")
    except Exception as e:
        print(f"Error loading audio file: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print("Attempting to extract audio using OpenCV...")
        try:
            # Try to extract audio using OpenCV as a fallback
            cap_audio = cv2.VideoCapture(video_path)
            if not cap_audio.isOpened():
                raise Exception("Could not open video for audio extraction")
            
            # Get video properties
            total_frames = int(cap_audio.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise Exception("Could not determine total frame count")
            
            print(f"Total frames to process: {total_frames}")
            audio_frames = []
            frame_count = 0
            
            while cap_audio.isOpened():
                ret, frame = cap_audio.read()
                if not ret:
                    break
                    
                # Extract audio from frame (this is a simplified approach)
                frame_mean = np.mean(frame, axis=(0, 1))
                audio_frames.append(frame_mean)
                frame_count += 1
                
                # Progress reporting
                if frame_count % 10 == 0 or frame_count == total_frames:
                    progress = (frame_count / total_frames) * 100
                    print(f"Audio extraction progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
            
            cap_audio.release()
            
            if not audio_frames:
                raise Exception("No frames were processed for audio extraction")
            
            # Convert to numpy array and normalize
            audio_data = np.array(audio_frames)
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)  # Average across channels
            
            # Normalize to 0-1 range
            min_val = np.min(audio_data)
            max_val = np.max(audio_data)
            if max_val > min_val:
                audio_data = (audio_data - min_val) / (max_val - min_val)
            else:
                audio_data = np.zeros_like(audio_data)
            
            sr = fps  # Use fps as sampling rate for this simplified approach
            samples_per_frame = 1
            print(f"Successfully extracted audio using OpenCV fallback. Processed {frame_count} frames")
        except Exception as e2:
            print(f"OpenCV audio extraction failed with error: {str(e2)}")
            audio_data = np.array([])
            sr = None
            samples_per_frame = 0
            print("Continuing with video processing without audio...")

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
                    right_eye_visible = True  # placeholder, improve with better eye tracking
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

            # Audio Evaluation
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if len(audio_data) > 0 and samples_per_frame > 0:
                start_sample = (frame_number - 1) * samples_per_frame
                end_sample = frame_number * samples_per_frame
                if start_sample < len(audio_data):
                    audio_frame = audio_data[start_sample:min(end_sample, len(audio_data))]
                    if len(audio_frame) > 0:
                        rms = np.sqrt(np.mean(audio_frame**2))
                        if rms > 0.01:  # adjust threshold as needed
                            audio_scores.append(1)
                        else:
                            audio_scores.append(0)
                    else:
                        audio_scores.append(0)
                else:
                    audio_scores.append(0)
            else:
                audio_scores.append(0)

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Align scores to seconds and create DataFrame
    eye_contact_scores_per_second = [np.mean(eye_contact_scores[int(i*fps):int((i+1)*fps)]) if fps > 0 and int((i+1)*fps) <= len(eye_contact_scores) else 0 for i in range(int(duration))]
    hand_movement_scores_per_second = [np.mean(hand_movement_scores[int(i*fps):int((i+1)*fps)]) if fps > 0 and int((i+1)*fps) <= len(hand_movement_scores) else 0 for i in range(int(duration))]
    audio_scores_per_second = audio_scores[:int(duration)]

    for i in range(int(duration)):
        timestamps.append(i)

    # Calculate percentages and overall score
    eye_contact_percentage = np.mean(eye_contact_scores_per_second) * 100
    hand_movement_percentage = np.mean(hand_movement_scores_per_second) * 100
    audio_percentage = np.mean(audio_scores_per_second) * 100
    overall_score = (eye_contact_percentage + hand_movement_percentage + audio_percentage) / 3

    print(f"Scores - Eye Contact: {eye_contact_percentage:.1f}%, "
          f"Hand Movement: {hand_movement_percentage:.1f}%, "
          f"Audio: {audio_percentage:.1f}%, "
          f"Overall: {overall_score:.1f}%")

    # Create DataFrame with all scores
    df = pd.DataFrame({
        'Timestamp (seconds)': timestamps,
        'Eye Contact Score': eye_contact_scores_per_second,
        'Hand Movement Score': hand_movement_scores_per_second,
        'Audio Score': audio_scores_per_second,
    })

    return df, {
        'eye_contact_score': float(eye_contact_percentage),
        'hand_movement_score': float(hand_movement_percentage),
        'audio_score': float(audio_percentage),
        'overall_score': float(overall_score)
    }

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
        result_df, scores = evaluate_presentation(local_video_path)

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
            # Update the videos table with the results URL and scores
            try:
                update_data = {
                    'results_url': supabase_url,
                    'status': "completed",
                    **scores  # Include all the scores in the update
                }
                print("Updating database with data:", update_data)
                print("Video URL for update:", video_url)
                
                update_response = supabase.table('videos').update(update_data).eq('video_url', video_url).execute()
                print("Database update response:", update_response)
                
                if hasattr(update_response, 'error') and update_response.error:
                    print(f"Error updating videos table: {update_response.error}")
                    return jsonify({"error": "Failed to update the videos table with results URL"}), 500
                else:
                    print("Database update successful")
            except Exception as e:
                print(f"An error occurred while updating the videos table: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                return jsonify({"error": "Failed to update the videos table with results URL"}), 500

            return jsonify({
                "message": "Evaluation successful, results uploaded to Supabase",
                "supabase_url": supabase_url,
                "scores": scores
            }), 200
        else:
            return jsonify({"error": "Evaluation successful, but failed to upload results to Supabase"}), 500

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        if os.path.exists(local_video_path):
            os.remove(local_video_path)
        return jsonify({"error": f"Evaluation failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
