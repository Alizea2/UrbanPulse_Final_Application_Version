import json
import re
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
# Local LLM used for all scoring and explanation calls, via Ollama.
MODEL_NAME = "qwen2.5:3b"

# Core score: sound + emotion, equally weighted.
SOUND_WEIGHT = 0.5
EMOTION_WEIGHT = 0.5

# Transcript and scene act as bounded modifiers, not equal partners:
# each can shift the core score by at most this much. 0.5 = no shift.
TRANSCRIPT_MODIFIER_MAX = 0.10
SCENE_MODIFIER_MAX = 0.10

SOUND_SCORE_RUBRIC = """0.0-0.2 = highly distressing (e.g. jackhammer, pile-driver, siren — impulsive, alarming, or industrial noise)
0.2-0.4 = moderately distressing (e.g. car horn, car alarm, large engine, shouting)
0.4-0.6 = mixed / neutral (e.g. large crowd, dog barking, amplified speech)
0.6-0.8 = moderately restorative (e.g. quiet talking, ambient music, small engine)
0.8-1.0 = highly restorative (e.g. birdsong, calm social ambiance, pleasant music)"""

EMOTION_SCORE_RUBRIC = """0.0-0.2 = highly distressing emotional state (e.g. Stressed)
0.2-0.4 = moderately distressing (e.g. Annoyed, Anxious)
0.4-0.6 = mixed / neutral
0.6-0.8 = moderately restorative
0.8-1.0 = highly restorative (e.g. Calm, Content)"""

# Rates sentiment of the words themselves, separate from vocal tone above.
TRANSCRIPT_SCORE_RUBRIC = """0.0-0.2 = clearly distressed/negative language (complaints, frustration, feeling overwhelmed)
0.2-0.4 = mildly negative language
0.4-0.6 = neutral/mixed language, no strong sentiment either way
0.6-0.8 = mildly positive language
0.8-1.0 = clearly positive/content language (enjoyment, calm, satisfaction)"""

# Green/natural scenes score high, industrial and traffic-dominated low.
SCENE_SCORE_RUBRIC = """0.0-0.2 = highly distressing scene (e.g. a construction site, an industrial area, a highway or motorway)
0.2-0.4 = moderately distressing scene (e.g. a busy street, a traffic intersection, a parking lot)
0.4-0.6 = mixed/neutral scene (e.g. a market or bazaar, a shopping mall, an office, inside a building)
0.6-0.8 = moderately restorative scene (e.g. a residential street, a café or restaurant, a quiet alley, a playground)
0.8-1.0 = highly restorative scene (e.g. a park, a garden, a forest or wooded area, a beach or waterfront)"""

# Stage 1a - "place" call: rates sound + scene in one request.
PLACE_PROMPT_WITH_SCENE = """You are the AI wellbeing engine inside UrbanPulse, an app that scores how a place affects a person's mental wellbeing.

You are given two independent readings of the PHYSICAL PLACE the user is in:
- The dominant ambient sound detected in the environment.
- The scene detected in a photo of the surroundings.

Sound label: {sound_label}
Scene: {scene_label}

Rate the SOUND's impact on wellbeing (0.0-1.0) using this rubric:
{sound_rubric}

Rate the SCENE's impact on wellbeing (0.0-1.0) using this rubric:
{scene_rubric}

Respond with ONLY a JSON object in exactly this format, no other text before or after it:
{{"sound_score": 0.00, "scene_score": 0.00}}
"""

# Stage 1b - "person" call: rates emotion (tone) + transcript (content).
PERSON_PROMPT_TEMPLATE = """You are the AI wellbeing engine inside UrbanPulse, an app that scores how a place affects a person's mental wellbeing.

You are given two independent readings of the PERSON's current state:
- Their emotional state, inferred from the TONE of a voice note they recorded.
- The TRANSCRIPT of that voice note — rate this on the sentiment of the WORDS THEMSELVES, separately from the emotion above. These can disagree (e.g. positive words said in a stressed tone) — rate each independently.

Emotion (from vocal tone): {emotion}
Voice note transcript: "{transcript}"

Rate the EMOTION's impact on wellbeing (0.0-1.0) using this rubric:
{emotion_rubric}

Rate the TRANSCRIPT's impact on wellbeing (0.0-1.0) using this rubric:
{transcript_rubric}

Respond with ONLY a JSON object in exactly this format, no other text before or after it:
{{"emotion_score": 0.00, "transcript_score": 0.00}}
"""

# Stage 2: the LLM is given the already-computed final score and asked to
# justify it, so the explanation can never disagree with the number shown.
EXPLANATION_PROMPT_TEMPLATE = """You are the AI wellbeing engine inside UrbanPulse, an app that scores how a place affects a person's mental wellbeing.

Sound label: {sound_label}
{scene_line}Emotion: {emotion}
Voice note transcript: "{transcript}"
Wellbeing Score: {wellbeing_score:.2f} (0.0 = highly distressing environment, 1.0 = highly restorative environment)

Write a one-sentence, plain-English explanation for why this environment received this Wellbeing Score. The explanation must reference both the sound and the emotion.

Respond with ONLY a JSON object in exactly this format, no other text before or after it:
{{"explanation": "one sentence here"}}
"""


DIARY_INSIGHT_PROMPT_TEMPLATE = """You are the AI wellbeing engine inside UrbanPulse. A user has logged the following diary entries, each capturing the dominant sound, their detected emotion, and the Wellbeing Score (0.0-1.0, higher = more restorative) for that moment:

{entries_block}

Write ONE short, encouraging sentence (max 30 words) pointing out a genuine pattern in this data — e.g. which sound or emotion tends to coincide with their best or worst scores. Only state a pattern the data actually supports; if the entries are too few or too mixed to support a real pattern, say that plainly instead of inventing one.

Respond with ONLY a JSON object in exactly this format, no other text before or after it:
{{"insight": "one sentence here"}}
"""


def generate_diary_insight(entries):
    """Summarizes diary entries (soundLabel, emotion, wellbeingScore) into
    one sentence."""
    lines = []
    for e in entries:
        sound = e.get("soundLabel") or "Unknown sound"
        emotion = (e.get("emotion") or {}).get("label") or "Unknown"
        score = e.get("wellbeingScore")
        if not isinstance(score, (int, float)):
            continue
        lines.append(f"- {sound} -> {emotion}, score {score:.2f}")

    if not lines:
        return "Save a few entries and your personalized insight will appear here."

    prompt = DIARY_INSIGHT_PROMPT_TEMPLATE.format(entries_block="\n".join(lines))
    raw = _call_ollama(prompt, temperature=0.4)
    parsed = _extract_json(raw)
    return str(parsed["insight"]).strip()


def _extract_json(raw_text):
    """Pulls the JSON object out of a response, in case the model adds prose."""
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model response: {raw_text!r}")
    return json.loads(match.group(0))


def _call_ollama(prompt, temperature):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
        },
        timeout=90,
    )
    response.raise_for_status()
    return response.json().get("response", "")


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def combine_scores(sound_score, emotion_score, transcript_score=None, scene_score=None, scene_confidence=None):
    """Fuses the rated sub-scores into the final score with a fixed formula.

    sound + emotion form the 50/50 core; transcript and scene each apply a
    bounded modifier. scene_confidence scales the scene modifier down when
    the classifier was unsure."""
    core = SOUND_WEIGHT * sound_score + EMOTION_WEIGHT * emotion_score

    modifier = 0.0
    if transcript_score is not None:
        modifier += (transcript_score - 0.5) * 2 * TRANSCRIPT_MODIFIER_MAX
    if scene_score is not None:
        confidence_weight = _clamp01(scene_confidence) if scene_confidence is not None else 1.0
        modifier += (scene_score - 0.5) * 2 * SCENE_MODIFIER_MAX * confidence_weight

    return round(_clamp01(core + modifier), 2)


def score_wellbeing(sound_label, emotion, transcript, scene_label, scene_confidence=None):
    """Returns (wellbeing_score, explanation) for the four input signals.

    1a. LLM rates sound + scene.  1b. LLM rates emotion + transcript.
    2. combine_scores() fuses them.  3. LLM explains the final score."""
    place_prompt = PLACE_PROMPT_WITH_SCENE.format(
        sound_label=sound_label, scene_label=scene_label,
        sound_rubric=SOUND_SCORE_RUBRIC, scene_rubric=SCENE_SCORE_RUBRIC,
    )

    raw_place = _call_ollama(place_prompt, temperature=0.0)
    parsed_place = _extract_json(raw_place)
    sound_score = _clamp01(parsed_place["sound_score"])
    scene_score = _clamp01(parsed_place["scene_score"])

    person_prompt = PERSON_PROMPT_TEMPLATE.format(
        emotion=emotion, transcript=transcript,
        emotion_rubric=EMOTION_SCORE_RUBRIC, transcript_rubric=TRANSCRIPT_SCORE_RUBRIC,
    )
    raw_person = _call_ollama(person_prompt, temperature=0.0)
    parsed_person = _extract_json(raw_person)
    emotion_score = _clamp01(parsed_person["emotion_score"])
    transcript_score = _clamp01(parsed_person["transcript_score"])

    wellbeing_score = combine_scores(sound_score, emotion_score, transcript_score, scene_score, scene_confidence)

    explanation_prompt = EXPLANATION_PROMPT_TEMPLATE.format(
        sound_label=sound_label, scene_line=f"Scene: {scene_label}\n", emotion=emotion, transcript=transcript,
        wellbeing_score=wellbeing_score,
    )
    raw_explanation = _call_ollama(explanation_prompt, temperature=0.4)
    parsed_explanation = _extract_json(raw_explanation)
    explanation = str(parsed_explanation["explanation"]).strip()

    return wellbeing_score, explanation
