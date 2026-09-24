from flask import Flask, request, jsonify
from flask_cors import CORS
from models.efficientat_classifier import load_efficientat_model, classify_audio
from models.transcriber import load_transcription_model, transcribe_audio
from models.emotion_classifier import load_emotion_model, classify_emotion_from_audio
from models.scene_classifier import load_scene_model, classify_scene_from_image
from models.wellbeing_scorer import score_wellbeing, generate_diary_insight
from models.sound_hunt import verify_challenge
from models import db
import io
from pydub import AudioSegment

app = Flask(__name__)
CORS(app)

# Server-side SQLite, so data is shared across clients rather than held
# in each device's local storage.
db.init_db()

print("Loading AI Models...")
try:
    # All four models are loaded once at startup and kept resident.
    efficientat_model = load_efficientat_model()
    transcription_model = load_transcription_model()
    emotion_model = load_emotion_model()
    scene_model = load_scene_model()
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    efficientat_model = None
    transcription_model = None
    emotion_model = None
    scene_model = None

def convert_to_wav_bytes(audio_bytes, sample_rate=16000):
    """Converts any incoming audio format to mono WAV at the given rate.

    Each route passes the rate its own model expects (32kHz / 16kHz)."""
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_frame_rate(sample_rate).set_channels(1)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        return wav_io.getvalue()
    except Exception as e:
        print(f"Format conversion skipped (using raw bytes): {e}")
        return audio_bytes

@app.route('/api/analyze', methods=['POST'])
def analyze_audio():
    """Classifies an uploaded audio clip into its top sound labels."""
    print("\n--- INCOMING ANALYSIS REQUEST ---")
    try:
        if request.is_json and 'audio' in request.json:
            print("Received base64 JSON payload, decoding...")
            import base64
            audio_bytes = base64.b64decode(request.json['audio'])
        elif 'audio' in request.files:
            file = request.files['audio']
            print(f"Received file: {file.filename}, reading bytes...")
            audio_bytes = file.read()
        else:
            print("ERROR: No audio provided")
            return jsonify({"error": "No audio file or base64 provided"}), 400

        if efficientat_model is None:
            return jsonify({"error": "Sound classification model failed to load"}), 500

        print("Converting bytes to WAV format...")
        # 32kHz matches EfficientAT's expected input rate.
        clean_wav_bytes = convert_to_wav_bytes(audio_bytes, sample_rate=32000)
        print("Passing to EfficientAT...")
        results = classify_audio(clean_wav_bytes, efficientat_model)
        print(f"EfficientAT Success: {results}")
        return jsonify({"status": "success", "results": results})
    except Exception as e:
        print(f"Analysis Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/scene', methods=['POST'])
def analyze_scene():
    """Classifies an uploaded photo into one of the candidate scene labels."""
    print("\n--- INCOMING SCENE REQUEST ---")
    try:
        if request.is_json and 'image' in request.json:
            print("Received base64 JSON payload, decoding...")
            import base64
            image_bytes = base64.b64decode(request.json['image'])
        elif 'image' in request.files:
            file = request.files['image']
            print(f"Received file: {file.filename}, reading bytes...")
            image_bytes = file.read()
        else:
            print("ERROR: No image provided")
            return jsonify({"error": "No image file or base64 provided"}), 400

        if scene_model is None:
            return jsonify({"error": "Scene classification model failed to load"}), 500

        print("Passing to CLIP...")
        result = classify_scene_from_image(image_bytes, scene_model)
        print(f"CLIP Success: {result['scene']} ({result['confidence']:.2f})")
        return jsonify({"status": "success", "scene": result["scene"], "confidence": result["confidence"]})
    except Exception as e:
        print(f"Scene Analysis Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    """Transcribes a voice note and classifies its emotion from the same audio."""
    print("\n--- INCOMING TRANSCRIPTION REQUEST ---")
    try:
        if request.is_json and 'audio' in request.json:
            print("Received base64 JSON payload, decoding...")
            import base64
            audio_bytes = base64.b64decode(request.json['audio'])
        elif 'audio' in request.files:
            file = request.files['audio']
            print(f"Received file: {file.filename}, reading bytes...")
            audio_bytes = file.read()
        else:
            print("ERROR: No audio provided")
            return jsonify({"error": "No audio file or base64 provided"}), 400
            
        if transcription_model is None:
            return jsonify({"error": "Transcription model failed to load"}), 500
        if emotion_model is None:
            return jsonify({"error": "Emotion detection model failed to load"}), 500

        # 16kHz is what wav2vec2 expects.
        clean_wav_bytes = convert_to_wav_bytes(audio_bytes)

        text = transcribe_audio(clean_wav_bytes, transcription_model)

        # Emotion is read from the audio itself, not from the transcript text.
        emotion_result = classify_emotion_from_audio(clean_wav_bytes, emotion_model)
        print(f"Emotion: {emotion_result['emotion']} (raw: {emotion_result['raw_label']})")

        return jsonify({
            "status": "success",
            "text": text,
            "emotion": emotion_result['emotion'],
            "emotion_confidence": emotion_result['confidence'],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/wellbeing', methods=['POST'])
def wellbeing():
    """Fuses sound, scene, transcript and emotion into a score and explanation."""
    print("\n--- INCOMING WELLBEING SCORE REQUEST ---")
    try:
        data = request.json or {}
        sound_label = data.get('sound_label')
        emotion = data.get('emotion')
        transcript = data.get('transcript')
        scene_label = data.get('scene_label')
        scene_confidence = data.get('scene_confidence')

        if not sound_label or not emotion or not transcript or not scene_label:
            return jsonify({"error": "sound_label, emotion, transcript and scene_label are required"}), 400

        print(f"Scoring wellbeing (sound={sound_label}, emotion={emotion}, scene={scene_label}, scene_confidence={scene_confidence})...")
        score, explanation = score_wellbeing(sound_label, emotion, transcript, scene_label, scene_confidence)
        print(f"Score: {score}, Explanation: {explanation}")

        return jsonify({
            "status": "success",
            "score": score,
            "explanation": explanation,
        })
    except Exception as e:
        print(f"Wellbeing Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/diary_insight', methods=['POST'])
def diary_insight():
    """Generates a one-sentence pattern insight across saved diary entries."""
    print("\n--- INCOMING DIARY INSIGHT REQUEST ---")
    try:
        data = request.json or {}
        entries = data.get('entries', [])

        insight = generate_diary_insight(entries)
        print(f"Insight: {insight}")

        return jsonify({"status": "success", "insight": insight})
    except Exception as e:
        print(f"Diary Insight Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/diary/entries', methods=['GET'])
def list_diary_entries():
    """Returns all diary entries, newest first."""
    try:
        return jsonify({"status": "success", "entries": db.get_diary_entries()})
    except Exception as e:
        print(f"List Diary Entries Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/diary/entries', methods=['POST'])
def create_diary_entry():
    """Saves a diary entry."""
    try:
        data = request.json or {}
        saved = db.save_diary_entry(data)
        return jsonify({"status": "success", "entry": saved})
    except Exception as e:
        print(f"Save Diary Entry Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/map/pins', methods=['GET'])
def list_map_pins():
    """Returns all map pins, newest first."""
    try:
        return jsonify({"status": "success", "pins": db.get_map_pins()})
    except Exception as e:
        print(f"List Map Pins Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/map/pins', methods=['POST'])
def create_map_pin():
    """Saves a map pin, replacing any existing pin within 80 metres."""
    try:
        data = request.json or {}
        saved = db.save_map_pin(data)
        return jsonify({"status": "success", "pin": saved})
    except Exception as e:
        print(f"Save Map Pin Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/sound_hunt/completions', methods=['GET'])
def list_sound_hunt_completions():
    """Returns the ids of completed Sound Hunt challenges."""
    try:
        return jsonify({"status": "success", "completed": db.get_completed_challenges()})
    except Exception as e:
        print(f"List Sound Hunt Completions Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/sound_hunt/complete', methods=['POST'])
def complete_sound_hunt_challenge():
    """Completes a challenge, re-classifying the submitted audio server-side
    and re-checking the keyword and time-window rules before accepting."""
    try:
        challenge_id = request.form.get('challenge_id') or (request.json or {}).get('challenge_id')
        if not challenge_id:
            return jsonify({"error": "challenge_id is required"}), 400

        audio_bytes = None
        if 'audio' in request.files:
            audio_bytes = request.files['audio'].read()
        elif request.is_json and 'audio' in request.json:
            import base64
            audio_bytes = base64.b64decode(request.json['audio'])

        if not audio_bytes:
            return jsonify({"error": "audio is required to verify challenge completion"}), 400

        if efficientat_model is None:
            return jsonify({"error": "Sound classification model failed to load"}), 500

        clean_wav_bytes = convert_to_wav_bytes(audio_bytes, sample_rate=32000)
        results = classify_audio(clean_wav_bytes, efficientat_model)
        if isinstance(results, dict):
            results = [results]

        ok, reason = verify_challenge(challenge_id, results)
        if not ok:
            print(f"Sound Hunt verification failed for '{challenge_id}': {reason}")
            return jsonify({"error": reason, "verified": False}), 400

        completed = db.mark_challenge_complete(challenge_id)
        return jsonify({"status": "success", "completed": completed, "verified": True})
    except Exception as e:
        print(f"Complete Sound Hunt Challenge Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run on all interfaces so the mobile device simulator can connect
    app.run(host='0.0.0.0', port=5001, debug=False)
