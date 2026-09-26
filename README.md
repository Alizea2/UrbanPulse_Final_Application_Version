# UrbanPulse

An urban soundscape wellbeing app. A user records a short audio clip and a photo
of where they are, optionally speaks a voice note, and receives a **Wellbeing
Score** (0.0–1.0) with a plain-English explanation of why that place scored the
way it did. Entries build a private diary and a shared community emotion map,
which in turn powers a Quiet Spot Finder and a Sound Hunt challenge mode.

A Flask backend runs five AI models locally. A React Native (Expo) client
provides the interface. No audio, photo or voice note leaves the machine the
backend is running on.

---

## The AI Pipeline

Five models run in sequence; the fifth fuses the outputs of the first four.

| Stage | Task | Model |
|---|---|---|
| 1 | Sound classification | EfficientAT `mn10_as` (vendored, MIT) |
| 1b | Scene classification | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` (zero-shot) |
| 2 | Speech transcription | `facebook/wav2vec2-large-960h-lv60-self` |
| 3 | Speech emotion recognition | `Yassmen/Wav2Vec2_Fine_tuned_on_CremaD_Speech_Emotion_Recognition` |
| 4 | Wellbeing scoring & explanation | `qwen2.5:3b` via Ollama |

Stage 4 makes two rated calls (place: sound + scene; person: emotion +
transcript), then `combine_scores()` fuses them with a **fixed, deterministic
formula** — sound and emotion form a 50/50 core, while transcript and scene each
apply a bounded ±0.10 modifier. The scene modifier is scaled by the classifier's
own confidence. The LLM never invents the final number; it is given the computed
score and asked to justify it.

---

## Folder Structure

```
UrbanPulse_Prototype/
├── api.py                      Flask API — 11 endpoints, loads all 5 models at startup
├── requirements.txt
├── test.wav                    Fixture clip used by the test suites
├── models/
│   ├── efficientat_classifier.py   Stage 1  — sound classification
│   ├── scene_classifier.py         Stage 1b — CLIP zero-shot scene
│   ├── transcriber.py              Stage 2  — wav2vec2 transcription
│   ├── emotion_classifier.py       Stage 3  — wav2vec2 SER
│   ├── wellbeing_scorer.py         Stage 4  — scoring + explanation
│   ├── sound_hunt.py               Server-side challenge verification
│   ├── db.py                       SQLite layer + geospatial deduplication
│   └── efficientat_src/            Vendored EfficientAT source (MIT)
├── mobile_app/
│   ├── src/screens/                Home, Analyse, Map, Diary, Quiet Spots, Sound Hunt
│   ├── src/components/             Shared UI (ScoreRing, SkylineBanner)
│   ├── src/theme/colors.js         Single source of truth for all colour tokens
│   └── src/utils/                  API clients for diary, map and hunt state
└── Test/
    ├── Unit Testing/               pytest (backend) + Jest (frontend)
    ├── Model Test/                 Model evaluation scripts, results, datasets
    ├── Functionality Testing/      End-to-end API tests
    └── Performance Testing/        Latency, concurrency, DB scaling, resources
```

---

## Setup Requirements

### System

| Requirement | Purpose |
|---|---|
| Python 3.13 | Backend and all AI stages |
| Node.js + npm | Mobile app and frontend tests |
| [ffmpeg](https://ffmpeg.org/) | Audio format conversion via `pydub` |
| [Ollama](https://ollama.com/) | Runs the Stage 4 language model locally |

Roughly **8 GB of RAM** is needed with all five models resident. Model weights
(~3 GB) download automatically on first run and are cached thereafter.

### Audio File Processing

Incoming audio is normalised to mono WAV before inference, at whichever sample
rate the target model expects — 32 kHz for sound classification, 16 kHz for
transcription and emotion. HEIC photos from iOS are decoded via `pillow-heif`,
which stock Pillow cannot read.

---

## Running the Application

### 1. Create virtual environment

```bash
python3 -m venv venv
```

### 2. Activate virtual environment

```bash
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull the language model

```bash
ollama pull qwen2.5:3b
```

Ollama must be running — the backend expects it at `http://localhost:11434`.

### 5. Run the Flask app

```bash
python3 -u api.py
```

Starts on `http://0.0.0.0:5001`. All five models load at startup; the first run
is slower while weights download. Wait for `Models loaded successfully.`

### 6. Run the mobile app

```bash
cd mobile_app
npm install
npx expo start
```

Press `i` for the iOS simulator or scan the QR code with Expo Go.

> **Testing on a physical device:** the client points at `http://127.0.0.1:5001`,
> which on a phone means the phone itself. Replace `API_BASE` with your
> machine's LAN IP (`ipconfig getifaddr en0`) in the six files under
> `mobile_app/src` that define it.

---

## API Endpoints

| Method | Route | Purpose |
|---|---|---|
| POST | `/api/analyze` | Classify ambient sound → top 3 labels |
| POST | `/api/scene` | Classify an environment photo |
| POST | `/api/transcribe` | Transcribe a voice note and detect its emotion |
| POST | `/api/wellbeing` | Fuse all four signals into a score + explanation |
| POST | `/api/diary_insight` | Generate a pattern insight across diary entries |
| GET / POST | `/api/diary/entries` | Read / save diary entries |
| GET / POST | `/api/map/pins` | Read / save community map pins |
| GET | `/api/sound_hunt/completions` | Completed challenge IDs |
| POST | `/api/sound_hunt/complete` | Submit a recording for server-side verification |

---

## Datasets

Three of the four evaluation datasets are **included** in this repository under
`Test/Model Test/Datasets/`, redistributed under their own licences. Full
citations are in [`Datasets/ATTRIBUTION.md`](Test/Model%20Test/Datasets/ATTRIBUTION.md).

| Dataset | Used for | Included? | Licence |
|---|---|---|---|
| SONYC-UST v2 (664-clip subset) | Sound classification (23 fine classes) | Yes | CC BY 4.0 |
| LibriSpeech `test-clean` | Speech transcription | Yes | CC BY 4.0 |
| RAVDESS (Actors 01–24) | Speech emotion recognition | Yes | CC BY-NC-SA 4.0 |
| Places365 (validation) | Scene classification | **No — fetch it yourself** | Non-commercial; **redistribution prohibited** |

> **RAVDESS is NonCommercial.** Because this repository includes it, the
> repository as a whole cannot be used for commercial purposes without first
> removing `Datasets/Emotion Detection from Speech/`.

**Places365** must be downloaded from
[places2.csail.mit.edu](http://places2.csail.mit.edu/); its licence forbids us
from shipping the images. Verify the current terms at each source before use.

### Pointing the scripts at a dataset

The three bundled datasets are already in place, but the scripts that use them
default to an absolute path from the original development machine, so pass
`--dataset-dir` to point at the copy in this repository:

```bash
python3 "Test/Model Test/model_evaluation/Sound Classification/test_efficientat_sonyc_ust.py" \
  --dataset-dir "Test/Model Test/Datasets/Sound Classification/SONYC-UST_test_split_664"
```

**Places365 is the exception** — the Scene Classification scripts take no
`--dataset-dir` and read from a fixed location instead. Place the 21 category
folders here:

```
Test/Model Test/Datasets/Scene Classification/places365_torch_uncertainty/
├── park/
├── botanical_garden/
├── forest-broadleaf/
├── beach/
├── street/
└── … 16 more (see SCENE_TO_PLACES365 in build_scene_eval_set.py)
```

Each folder needs at least 20 images. Then build the eval set before testing —
this writes `scene_eval_set.csv` (21 scenes × 20 = 420 rows) alongside the
images, which `test_scene_classification.py` then reads:

```bash
python3 "Test/Model Test/model_evaluation/Scene Classification/build_scene_eval_set.py"
```

The Stage 4 evaluation set is generated rather than downloaded, and is built by
`build_wellbeing_eval_set_4input.py` under
`model_evaluation/Wellbeing Scoring & Explanation/range testing/`.

---

## Running Tests

Four independent test suites, each answering a different question.

### 1. Unit Tests

*Is each deterministic piece correct in isolation?* Runs in under a second; the
AI models are stubbed out, so no weights are loaded.

```bash
# Backend — 30 tests (API routes, database, scoring arithmetic)
python3 -m pytest "Test/Unit Testing/backend" -v
```

```bash
# Frontend — 21 tests (colour gradient, mood bands, Sound Hunt store)
cd "Test/Unit Testing/frontend" && npx jest
```

### 2. Model Tests

*Which model should each pipeline stage use?* Benchmarks candidate models for
every stage against a public dataset, then merges the results into one
comparison. Requires the datasets above, and takes a long time.

> **One extra prerequisite.** `test_efficientat_sonyc_ust.py` benchmarks the
> upstream EfficientAT training code, so it needs a clone of the original
> repository at `/tmp/EfficientAT`:
>
> ```bash
> git clone https://github.com/fschmid56/EfficientAT /tmp/EfficientAT
> ```
>
> This is only needed for that one evaluation script. **The application itself
> needs nothing extra** — it uses the copy vendored at `models/efficientat_src/`.
> To clone elsewhere, edit `EFFICIENTAT_REPO` at the top of the script.

```bash
cd "Test/Model Test/model_evaluation"
python3 "Sound Classification/test_efficientat_sonyc_ust.py" --dataset-dir /path/to/SONYC
python3 "Sound Classification/compare_sonyc_ust_models.py"
```

The same test-then-compare pattern applies to Voice Note Transcription, Emotion
Detection from Speech, and Wellbeing Scoring & Explanation. Scene Classification
follows it too, but takes no `--dataset-dir` and needs its eval set built first
— see the Places365 note above. Results and comparison charts are written to
`Test/Model Test/Test-Results/`.

### 3. Functionality Tests

*Does every feature work end to end over real HTTP?* Twelve tests covering all
backend features, including invalid input and backend-unreachable handling.
**The Flask server must already be running.**

```bash
python3 "Test/Functionality Testing/functionality_test.py"
```

### 4. Performance Tests

*Is it fast enough, and does it hold up under load?* Also requires a running
server with models warm-loaded.

```bash
# Per-stage and end-to-end latency (n=20, warm-up runs excluded)
python3 "Test/Performance Testing/latency_test.py"

# Skip the two stages that call the LLM
python3 "Test/Performance Testing/latency_test.py" --steps13-only

# Behaviour under 1, 2, 4 and 8 simultaneous requests
python3 "Test/Performance Testing/concurrency_test.py"

# Query latency as the tables grow to 10k+ rows
python3 "Test/Performance Testing/db_scaling_test.py"

# Sample memory and CPU of the running backend for 90s
python3 "Test/Performance Testing/resource_monitor.py" 90
```

---

## Notes and Limitations

- **Everything runs locally.** Audio and photos are processed on the backend
  machine and never sent to a third-party API.
- **The API has no authentication.** It is a prototype intended for local and
  LAN use, not public deployment.
- **Diary and map data are shared.** All clients talking to one backend read and
  write the same SQLite database, so the map is genuinely communal — but diary
  entries are not private between users of the same server.
- **Scene classification is the weakest stage** (67.1% top-1), which is why its
  contribution to the final score is both bounded and confidence-scaled.
