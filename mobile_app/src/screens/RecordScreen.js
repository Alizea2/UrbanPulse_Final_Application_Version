import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ActivityIndicator, ScrollView, Dimensions, Alert, Image, SafeAreaView } from 'react-native';
import * as Haptics from 'expo-haptics';
import * as Location from 'expo-location';
import * as ImagePicker from 'expo-image-picker';
import { useAudioRecorder, AudioModule, setAudioModeAsync, IOSOutputFormat, AudioQuality } from 'expo-audio';
import * as FileSystem from 'expo-file-system/legacy';
import { FileSystemUploadType } from 'expo-file-system/legacy';
import Animated, { useSharedValue, useAnimatedStyle, withRepeat, withTiming, withSequence } from 'react-native-reanimated';
import { Feather } from '@expo/vector-icons';
import { COLORS, EMOTION_COLORS, getScoreColor, getScoreMood } from '../theme/colors';
import { saveDiaryEntry } from '../utils/diaryStore';
import { saveMapPin } from '../utils/mapStore';
import ScoreRing from '../components/ScoreRing';
import SkylineBanner from '../components/SkylineBanner';

const { width, height } = Dimensions.get('window');

const API_BASE = "http://127.0.0.1:5001";

const WAV_RECORDING_OPTIONS = {
  extension: '.wav',
  sampleRate: 16000,
  numberOfChannels: 1,
  bitRate: 128000,
  ios: {
    outputFormat: IOSOutputFormat.LINEARPCM,
    audioQuality: AudioQuality.MAX,
    linearPCMBitDepth: 16,
    linearPCMIsBigEndian: false,
    linearPCMIsFloat: false,
  },
  android: {
    extension: '.wav',
    outputFormat: 'default',
    audioEncoder: 'default',
  },
  web: {
    mimeType: 'audio/webm',
  }
};


// The 4-stage pipeline shown as a persistent stepper at the top of the
// screen — one glance tells the user where they are, instead of a wall
// of instructional text per section. Mirrors Home's "How it works" cards.
const PIPELINE = [
  { key: 'listen', icon: 'volume-2', steps: ['IDLE', 'RECORDING_ENV', 'PROCESSING_ENV'] },
  { key: 'see', icon: 'camera', steps: ['ASK_PHOTO', 'PROCESSING_PHOTO'] },
  { key: 'feel', icon: 'mic', steps: ['ASK_VOICE', 'RECORDING_VOICE', 'PROCESSING_VOICE'] },
  { key: 'score', icon: 'activity', steps: ['PROCESSING_SCORE', 'DONE'] },
];

function StepIndicator({ step }) {
  const activeIndex = PIPELINE.findIndex((p) => p.steps.includes(step));
  return (
    <View style={styles.stepperRow}>
      {PIPELINE.map((p, i) => {
        const isDone = i < activeIndex;
        const isActive = i === activeIndex;
        const color = isDone || isActive ? COLORS.primary : COLORS.textMuted;
        return (
          <React.Fragment key={p.key}>
            <View style={[
              styles.stepDot,
              isActive && styles.stepDotActive,
              isDone && styles.stepDotDone,
            ]}>
              <Feather name={isDone ? 'check' : p.icon} size={14} color={isDone || isActive ? COLORS.onPrimary : color} />
            </View>
            {i < PIPELINE.length - 1 && <View style={[styles.stepLine, isDone && styles.stepLineDone]} />}
          </React.Fragment>
        );
      })}
    </View>
  );
}

export default function RecordScreen({ navigation }) {
  const [step, setStep] = useState('IDLE'); // IDLE, RECORDING_ENV, PROCESSING_ENV, ASK_PHOTO, PROCESSING_PHOTO, ASK_VOICE, RECORDING_VOICE, PROCESSING_VOICE, PROCESSING_SCORE, DONE

  const audioRecorder = useAudioRecorder({
    ...WAV_RECORDING_OPTIONS,
    isMeteringEnabled: true,
  });

  const [envResults, setEnvResults] = useState(null);
  const [envPhotoUri, setEnvPhotoUri] = useState(null); // local URI of the captured/picked environment photo
  const [sceneResults, setSceneResults] = useState(null); // [{label, confidence}, ...] from CLIP, top result first
  const [transcript, setTranscript] = useState(null);
  const [emotion, setEmotion] = useState(null); // { label, confidence } — from GoEmotions in Step 2
  const [wellbeing, setWellbeing] = useState(null); // { score, explanation } — from the LLM in Step 3
  const [errorMsg, setErrorMsg] = useState(null);
  const [entrySaved, setEntrySaved] = useState(false);
  const [mapPin, setMapPin] = useState(null); // set once "Submit to Map" succeeds; button becomes "Go to Map"

  const [currentDb, setCurrentDb] = useState(0);

  useEffect(() => {
    let interval;
    if (step === 'RECORDING_ENV' || step === 'RECORDING_VOICE') {
      interval = setInterval(() => {
        if (audioRecorder && audioRecorder.status && audioRecorder.status.metering !== undefined) {
          const db = audioRecorder.status.metering;
          // map -60 to 0 to an amplitude multiplier of 0 to 1
          const normalized = Math.max(0, Math.min(1, (db + 60) / 60));
          setCurrentDb(normalized);
        } else {
          setCurrentDb(Math.random() * 0.5);
        }
      }, 50);
    } else {
      setCurrentDb(0);
    }
    return () => clearInterval(interval);
  }, [step, audioRecorder]);

  // The beautiful glowing Siri Wave component
  const renderSiriWave = (amplitude) => {
    const time = Date.now() / 200; // Continuous smooth phase

    const renderLayer = (color, maxH, speed, waveCount) => (
      <View style={{ flexDirection: 'row', position: 'absolute', alignItems: 'center' }}>
        {Array.from({ length: 45 }).map((_, i) => {
          const x = (i / 22) - 1; // -1 to 1
          const bell = Math.exp(-Math.pow(x, 2) * 3); // Bell curve to pinch the ends
          const sine = Math.abs(Math.sin((x * waveCount) + time * speed)); // Wavy motion
          const height = 4 + (bell * sine * (amplitude * maxH)); // Minimum height is 4

          return (
            <View
              key={i}
              style={{
                width: 4, height: height, backgroundColor: color,
                borderRadius: 4, marginHorizontal: 1, opacity: 0.7,
                shadowColor: color, shadowOpacity: 1, shadowRadius: 10,
              }}
            />
          );
        })}
      </View>
    );

    return (
      <View style={styles.waveformRow}>
        {renderLayer(COLORS.secondary, 120, 1.2, 3)}
        {renderLayer('#C97B6B', 80, 0.8, 2)}
        {renderLayer(COLORS.primary, 100, 1.5, 4)}
      </View>
    );
  };


  const isProcessingRef = React.useRef(false);

  const stopRecordingAndUpload = async (isEnv) => {
    if (isProcessingRef.current) return; // Prevent double taps!
    isProcessingRef.current = true;

    try {
      setStep(isEnv ? 'PROCESSING_ENV' : 'PROCESSING_VOICE');

      // Stop recording safely
      try {
        await audioRecorder.stop();
      } catch (stopErr) {
        // If it's already stopped, ignore
      }

      await new Promise(resolve => setTimeout(resolve, 500)); // allow flush

      let fileUri = audioRecorder.uri;
      if (!fileUri) throw new Error("No recording URI found");
      if (!fileUri.startsWith('file://')) fileUri = 'file://' + fileUri;

      // Check if file actually exists (Simulator mic issues often result in 0 byte deleted files)
      const fileInfo = await FileSystem.getInfoAsync(fileUri);
      if (!fileInfo.exists) {
        throw new Error("Audio file was not saved. Your simulator microphone might not be picking up sound. Please check Mac sound settings.");
      }

      // decide which endpoint to call based on what we're uploading
      const endpoint = isEnv ? '/api/analyze' : '/api/transcribe';
      console.log(`Uploading to ${API_BASE}${endpoint}...`);

      // convert the audio file to base64 so we can send it as JSON
      const base64Audio = await FileSystem.readAsStringAsync(fileUri, { encoding: FileSystem.EncodingType.Base64 });

      // send the audio to the Flask backend
      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({ audio: base64Audio })
      });

      // if something went wrong on the server, throw an error
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server returned ${response.status}: ${errorText}`);
      }

      const data = await response.json();
      if (data.error) throw new Error(data.error);

      if (isEnv) {
        // sound classification done — save results and move to the environment
        // photo step (feeds a scene label into Step 4 alongside the sound label)
        setEnvResults(data.results);
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
        setStep('ASK_PHOTO');
      } else {
        // transcription + GoEmotions done — save transcript and emotion, then fuse everything into a wellbeing score
        setTranscript(data.text);
        setEmotion({ label: data.emotion, confidence: data.emotion_confidence });
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
        setStep('PROCESSING_SCORE');
        fetchWellbeingScore(data.text, data.emotion);
      }
    } catch (err) {
      console.error(err);
      setErrorMsg(err.message || 'Upload failed.');
      setStep(isEnv ? 'IDLE' : 'ASK_VOICE');
    } finally {
      isProcessingRef.current = false;
    }
  };

  const fetchWellbeingScore = async (transcriptText, emotionLabel) => {
    try {
      const soundLabel = envResults && envResults.length > 0 ? envResults[0].label : 'Unknown';
      const sceneLabel = sceneResults && sceneResults.length > 0 ? sceneResults[0].label : null;
      const sceneConfidence = sceneResults && sceneResults.length > 0 ? sceneResults[0].confidence : null;
      console.log(`Requesting wellbeing score for sound="${soundLabel}", emotion="${emotionLabel}", scene="${sceneLabel}" (confidence=${sceneConfidence})...`);

      const response = await fetch(`${API_BASE}/api/wellbeing`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({
          sound_label: soundLabel,
          emotion: emotionLabel,
          transcript: transcriptText,
          scene_label: sceneLabel,
          scene_confidence: sceneConfidence,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server returned ${response.status}: ${errorText}`);
      }

      const data = await response.json();
      if (data.error) throw new Error(data.error);

      setWellbeing({ score: data.score, explanation: data.explanation });
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    } catch (err) {
      console.error(err);
      setErrorMsg(err.message || 'Wellbeing scoring failed.');
      setWellbeing(null);
    } finally {
      setStep('DONE');
    }
  };

  const handleSaveDiaryEntry = async () => {
    try {
      const soundLabel = envResults && envResults.length > 0 ? envResults[0].label : 'Unknown';
      const sceneLabel = sceneResults && sceneResults.length > 0 ? sceneResults[0].label : null;
      await saveDiaryEntry({
        soundLabel,
        sceneLabel,
        transcript,
        emotion,
        wellbeingScore: wellbeing ? wellbeing.score : null,
        wellbeingExplanation: wellbeing ? wellbeing.explanation : null,
      });

      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      Alert.alert('Saved', 'This entry has been added to your Mood Diary.');
      setEntrySaved(true);
    } catch (err) {
      console.error(err);
      Alert.alert('Save Failed', err.message || 'Could not save this diary entry.');
    }
  };

  // Resets the whole pipeline back to the start so the user can log a
  // fresh Pulse without leaving the screen — same reset as tapping the
  // mic from IDLE/DONE, but doesn't jump straight into recording.
  const handleReanalyse = () => {
    setStep('IDLE');
    setErrorMsg('');
    setEnvResults(null);
    setEnvPhotoUri(null);
    setSceneResults(null);
    setTranscript(null);
    setEmotion(null);
    setWellbeing(null);
    setEntrySaved(false);
    setMapPin(null);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };

  const submitToMap = async () => {
    // Once already submitted, this button becomes "Go to Map" — just navigate,
    // don't re-fetch location or create a duplicate pin.
    if (mapPin) {
      navigation.navigate('Map');
      return;
    }

    try {
      const perm = await Location.requestForegroundPermissionsAsync();
      if (perm.status !== 'granted') {
        Alert.alert('Location Needed', 'Location access is required to place this report on the community map.');
        return;
      }

      // Default accuracy (Balanced) trades precision for speed/battery and can be
      // off by a significant margin — this pin's coordinates are what actually
      // gets shown to the community, so it needs the best fix available.
      const position = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.BestForNavigation,
      });
      const soundLabel = envResults && envResults.length > 0 ? envResults[0].label : 'Unknown sound';
      const sceneLabel = sceneResults && sceneResults.length > 0 ? sceneResults[0].label : null;
      const emotionLabel = emotion ? emotion.label : 'Calm';

      const pin = {
        id: String(Date.now()),
        loc: { latitude: position.coords.latitude, longitude: position.coords.longitude },
        emotion: emotionLabel,
        desc: soundLabel,
        sceneLabel,
        color: EMOTION_COLORS[emotionLabel]?.color || COLORS.textMuted,
        wellbeingScore: wellbeing ? wellbeing.score : null,
      };

      await saveMapPin(pin);
      setMapPin(pin);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      Alert.alert('Submitted', 'Your report has been added to the community map.');
    } catch (err) {
      console.error(err);
      Alert.alert('Submit Failed', err.message || 'Could not get your location.');
    }
  };

  const classifyScenePhoto = async (uri) => {
    setStep('PROCESSING_PHOTO');
    try {
      const base64Image = await FileSystem.readAsStringAsync(uri, { encoding: FileSystem.EncodingType.Base64 });

      const response = await fetch(`${API_BASE}/api/scene`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: JSON.stringify({ image: base64Image }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server returned ${response.status}: ${errorText}`);
      }

      const data = await response.json();
      if (data.error) throw new Error(data.error);

      setSceneResults([{ label: data.scene, confidence: data.confidence }]);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    } catch (err) {
      console.error('Scene classification error:', err);
      setErrorMsg(err.message || 'Scene classification failed.');
      setEnvPhotoUri(null);
    } finally {
      setStep('ASK_PHOTO');
    }
  };

  const handleTakePhoto = async () => {
    try {
      const perm = await ImagePicker.requestCameraPermissionsAsync();
      if (perm.status !== 'granted') {
        Alert.alert('Camera Needed', 'Camera access is required to photograph your surroundings.');
        return;
      }
      const result = await ImagePicker.launchCameraAsync({ quality: 0.6 });
      if (!result.canceled && result.assets && result.assets.length > 0) {
        setEnvPhotoUri(result.assets[0].uri);
        await classifyScenePhoto(result.assets[0].uri);
      }
    } catch (err) {
      console.error('Take photo error:', err);
      Alert.alert('Camera Failed', err.message || 'Could not open the camera.');
    }
  };

  const handlePickPhoto = async () => {
    try {
      const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (perm.status !== 'granted') {
        Alert.alert('Photo Library Needed', 'Photo library access is required to choose a photo.');
        return;
      }
      const result = await ImagePicker.launchImageLibraryAsync({ quality: 0.6 });
      if (!result.canceled && result.assets && result.assets.length > 0) {
        setEnvPhotoUri(result.assets[0].uri);
        await classifyScenePhoto(result.assets[0].uri);
      }
    } catch (err) {
      console.error('Pick photo error:', err);
      Alert.alert('Selection Failed', err.message || 'Could not open the photo library.');
    }
  };

  const handleMicPress = async () => {
    try {
      if (step === 'IDLE' || step === 'DONE') {
        setErrorMsg('');
        setWellbeing(null);
        setEmotion(null);
        setEntrySaved(false);
        setMapPin(null);
        setEnvPhotoUri(null);
        setSceneResults(null);
        const perm = await AudioModule.requestRecordingPermissionsAsync();
        if (perm.status !== 'granted') {
          throw new Error("Permission not granted");
        }
        await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });

        // Explicitly prepare the recorder for a new session
        await audioRecorder.prepareToRecordAsync({
          ...WAV_RECORDING_OPTIONS,
          isMeteringEnabled: true,
        });

        audioRecorder.record();
        setStep('RECORDING_ENV');
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

      } else if (step === 'RECORDING_ENV') {
        stopRecordingAndUpload(true);
      } else if (step === 'ASK_VOICE') {
        setErrorMsg('');

        // Prepare recorder for voice note
        await audioRecorder.prepareToRecordAsync({
          ...WAV_RECORDING_OPTIONS,
          isMeteringEnabled: true,
        });

        audioRecorder.record();
        setStep('RECORDING_VOICE');
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
      } else if (step === 'RECORDING_VOICE') {
        stopRecordingAndUpload(false);
      }
    } catch (err) {
      console.error("Mic Error:", err);
      setErrorMsg(err.message || "Failed to start microphone.");
    }
  };

  return (
    <SafeAreaView style={styles.screen}>
    <ScrollView contentContainerStyle={styles.scrollContainer} showsVerticalScrollIndicator={false}>
      <View style={styles.bannerBleed}>
        <SkylineBanner />
      </View>

      <StepIndicator step={step} />

      {errorMsg ? (
        <View style={styles.errorBanner}>
          <Feather name="alert-circle" size={14} color={COLORS.danger} />
          <Text style={styles.errorBannerText} numberOfLines={2}>{errorMsg}</Text>
        </View>
      ) : null}

      {/* RECORD — ambient sound */}
      <View style={styles.section}>
        <Text style={styles.sectionLabel}>RECORD</Text>

        {step === 'IDLE' || step === 'RECORDING_ENV' || step === 'PROCESSING_ENV' ? (
          <View style={styles.waveContainer}>
            {step === 'IDLE' && (
              <Text style={styles.recordPrompt}>Record your current environment soundscape</Text>
            )}

            <View style={styles.haloOuter}>
              <View style={styles.haloInner}>
                <TouchableOpacity
                  style={[styles.recordButton, step === 'RECORDING_ENV' && styles.recordButtonActive]}
                  onPress={handleMicPress}
                  disabled={step === 'PROCESSING_ENV'}
                >
                  {step === 'PROCESSING_ENV' ? (
                    <ActivityIndicator size="large" color="#FFF" />
                  ) : (
                    step === 'RECORDING_ENV' ? <Feather name="square" size={32} color="#FFF" /> : <Feather name="mic" size={32} color={COLORS.onPrimary} />
                  )}
                </TouchableOpacity>
              </View>
            </View>

            <Text style={styles.statusText}>
              {step === 'IDLE' ? 'Tap to record' : step === 'RECORDING_ENV' ? 'Recording…' : 'Analysing…'}
            </Text>

            <View style={styles.waveformWrapper}>
              {(step === 'IDLE' || step === 'RECORDING_ENV') && renderSiriWave(currentDb)}
            </View>
          </View>
        ) : (
          <View style={styles.card}>
            {envResults && envResults.length > 0 ? (
              <>
                {envResults[0]?.low_confidence && (
                  <View style={styles.lowConfidenceRow}>
                    <Feather name="alert-triangle" size={12} color={COLORS.warning} />
                    <Text style={styles.lowConfidenceNote}>Ambiguous sound — low confidence</Text>
                  </View>
                )}
                <View style={styles.tagContainer}>
                  {envResults.map((result, index) => (
                    <View key={index} style={[styles.soundTag, index === 0 && !result.low_confidence && styles.dominantTag]}>
                      <Text style={[styles.soundTagText, index === 0 && !result.low_confidence && styles.dominantTagText]}>
                        {result.label} · {Math.round(result.confidence * 100)}%
                      </Text>
                    </View>
                  ))}
                </View>
              </>
            ) : (
              <Text style={styles.cardBody}>No results</Text>
            )}
          </View>
        )}
      </View>

      {/* SEE — environment photo, feeds a scene label into the score alongside the sound label.
          Unlike the old version, this stays visible for the rest of the flow (matching LISTEN)
          instead of disappearing once the user moves on to FEEL/SCORE — the photo and scene
          label are a core part of what built the final score, so they should stay on screen. */}
      {step !== 'IDLE' && step !== 'RECORDING_ENV' && step !== 'PROCESSING_ENV' && (
        <View style={styles.section}>
          <Text style={styles.sectionLabel}>SEE</Text>

          {step === 'PROCESSING_PHOTO' ? (
            <View style={styles.waveContainer}>
              <ActivityIndicator size="large" color={COLORS.primary} />
              <Text style={styles.statusText}>Reading the scene…</Text>
            </View>
          ) : sceneResults ? (
            <View style={styles.card}>
              {envPhotoUri && <Image source={{ uri: envPhotoUri }} style={styles.photoPreview} />}
              <View style={styles.tagContainer}>
                {sceneResults.map((result, index) => (
                  <View key={index} style={[styles.soundTag, index === 0 && styles.dominantTag]}>
                    <Text style={[styles.soundTagText, index === 0 && styles.dominantTagText]}>
                      {result.label} · {Math.round(result.confidence * 100)}%
                    </Text>
                  </View>
                ))}
              </View>
              {step === 'ASK_PHOTO' && (
                <>
                  <TouchableOpacity style={styles.submitButton} onPress={() => setStep('ASK_VOICE')}>
                    <Feather name="arrow-right" size={16} color={COLORS.onPrimary} style={{ marginRight: 6 }} />
                    <Text style={styles.submitButtonText}>Continue</Text>
                  </TouchableOpacity>
                  <TouchableOpacity onPress={() => { setEnvPhotoUri(null); setSceneResults(null); }}>
                    <Text style={styles.skipText}>Retake</Text>
                  </TouchableOpacity>
                </>
              )}
            </View>
          ) : (
            <View style={styles.card}>
              <View style={styles.iconPrompt}>
                <Feather name="map-pin" size={20} color={COLORS.textMuted} />
                <Text style={styles.iconPromptText}>What's around you?</Text>
              </View>
              <TouchableOpacity style={styles.submitButton} onPress={handleTakePhoto}>
                <Feather name="camera" size={16} color={COLORS.onPrimary} style={{ marginRight: 6 }} />
                <Text style={styles.submitButtonText}>Take Photo</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[styles.submitButton, styles.secondaryButton]} onPress={handlePickPhoto}>
                <Feather name="image" size={16} color={COLORS.primary} style={{ marginRight: 6 }} />
                <Text style={[styles.submitButtonText, styles.secondaryButtonText]}>Choose from Library</Text>
              </TouchableOpacity>
            </View>
          )}
        </View>
      )}

      {/* FEEL — voice note */}
      {(step === 'ASK_VOICE' || step === 'RECORDING_VOICE' || step === 'PROCESSING_VOICE' || step === 'PROCESSING_SCORE' || step === 'DONE') && (
        <View style={styles.section}>
          <Text style={styles.sectionLabel}>FEEL</Text>

          {step === 'ASK_VOICE' || step === 'RECORDING_VOICE' || step === 'PROCESSING_VOICE' ? (
            <View style={styles.waveContainer}>
              <View style={styles.haloOuter}>
                <View style={styles.haloInner}>
                  <TouchableOpacity
                    style={[styles.recordButton, step === 'RECORDING_VOICE' && styles.recordButtonActive]}
                    onPress={handleMicPress}
                    disabled={step === 'PROCESSING_VOICE'}
                  >
                    {step === 'PROCESSING_VOICE' ? (
                      <ActivityIndicator size="large" color="#FFF" />
                    ) : (
                      step === 'RECORDING_VOICE' ? <Feather name="square" size={32} color="#FFF" /> : <Feather name="mic" size={32} color={COLORS.onPrimary} />
                    )}
                  </TouchableOpacity>
                </View>
              </View>

              <Text style={styles.statusText}>
                {step === 'ASK_VOICE' ? 'Tap to speak' : step === 'RECORDING_VOICE' ? 'Listening…' : 'Transcribing…'}
              </Text>

              <View style={styles.waveformWrapper}>
                {(step === 'ASK_VOICE' || step === 'RECORDING_VOICE') && renderSiriWave(currentDb)}
              </View>
            </View>
          ) : (
            <View style={styles.card}>
              <Text style={styles.cardBody}>{transcript || "No transcript available."}</Text>

              {emotion && (
                <View style={[styles.emotionBadge, { borderColor: EMOTION_COLORS[emotion.label]?.color || COLORS.textMuted, backgroundColor: `${EMOTION_COLORS[emotion.label]?.color || COLORS.textMuted}22` }]}>
                  <Text style={[styles.emotionBadgeText, { color: EMOTION_COLORS[emotion.label]?.color || COLORS.textMuted }]}>
                    {emotion.label} · {Math.round(emotion.confidence * 100)}%
                  </Text>
                </View>
              )}

              <TouchableOpacity
                style={[styles.submitButton, (step === 'PROCESSING_SCORE' || entrySaved) && styles.submitButtonDisabled]}
                onPress={handleSaveDiaryEntry}
                disabled={step === 'PROCESSING_SCORE' || entrySaved}
              >
                <Feather name={entrySaved ? 'check' : 'save'} size={16} color={COLORS.onPrimary} style={{ marginRight: 6 }} />
                <Text style={styles.submitButtonText}>
                  {step === 'PROCESSING_SCORE' ? 'Scoring…' : entrySaved ? 'Saved' : 'Save to Diary'}
                </Text>
              </TouchableOpacity>
            </View>
          )}
        </View>
      )}

      {/* SCORE — local LLM fusing sound + scene + emotion + transcript */}
      {(step === 'PROCESSING_SCORE' || step === 'DONE') && (
        <View style={styles.section}>
          <Text style={styles.sectionLabel}>SCORE</Text>

          {step === 'PROCESSING_SCORE' ? (
            <View style={styles.waveContainer}>
              <ActivityIndicator size="large" color={COLORS.primary} />
              <Text style={styles.statusText}>Building your Pulse…</Text>
            </View>
          ) : wellbeing ? (
            <View style={styles.scoreWrap}>
              <ScoreRing score={wellbeing.score} size={200} strokeWidth={16} />

              <Text style={styles.explanationText}>{wellbeing.explanation}</Text>

              <TouchableOpacity style={styles.submitButton} onPress={submitToMap}>
                <Feather name="map-pin" size={16} color={COLORS.onPrimary} style={{ marginRight: 6 }} />
                <Text style={styles.submitButtonText}>{mapPin ? 'Go to Map' : 'Submit to Map'}</Text>
              </TouchableOpacity>

              <TouchableOpacity style={[styles.submitButton, styles.secondaryButton]} onPress={handleReanalyse}>
                <Feather name="refresh-cw" size={16} color={COLORS.primary} style={{ marginRight: 6 }} />
                <Text style={[styles.submitButtonText, styles.secondaryButtonText]}>Re-analyse</Text>
              </TouchableOpacity>
            </View>
          ) : (
            <View style={styles.card}>
              <View style={styles.iconPrompt}>
                <Feather name="alert-circle" size={20} color={COLORS.danger} />
                <Text style={styles.iconPromptText}>{errorMsg || "Couldn't score this Pulse."}</Text>
              </View>
            </View>
          )}
        </View>
      )}

    </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: COLORS.background },
  scrollContainer: { flexGrow: 1, alignItems: 'center', paddingTop: 0, paddingBottom: 50 },
  bannerBleed: { marginBottom: 10 },
  stepperRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 28,
  },
  stepDot: {
    width: 32,
    height: 32,
    borderRadius: 16,
    borderWidth: 1.5,
    borderColor: COLORS.cardBorder,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stepDotActive: {
    borderColor: COLORS.primary,
    backgroundColor: COLORS.primary,
  },
  stepDotDone: {
    borderColor: COLORS.primary,
    backgroundColor: COLORS.primary,
  },
  stepLine: {
    width: 26,
    height: 1.5,
    backgroundColor: COLORS.cardBorder,
    marginHorizontal: 4,
  },
  stepLineDone: {
    backgroundColor: COLORS.primary,
  },
  errorBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    width: '92%',
    backgroundColor: 'rgba(184, 74, 58, 0.1)',
    borderWidth: 1,
    borderColor: 'rgba(184, 74, 58, 0.3)',
    borderRadius: 12,
    padding: 10,
    marginBottom: 20,
  },
  errorBannerText: {
    color: COLORS.danger,
    fontSize: 12.5,
    marginLeft: 8,
    flex: 1,
  },
  section: { width: '92%', marginBottom: 30, alignItems: 'center' },
  sectionLabel: {
    color: COLORS.secondary,
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 1.5,
    alignSelf: 'flex-start',
    marginBottom: 14,
  },
  waveContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: COLORS.surface,
    borderRadius: 24,
    paddingVertical: 36,
    paddingHorizontal: 20,
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
    width: '100%',
  },
  haloOuter: {
    width: 140,
    height: 140,
    borderRadius: 70,
    backgroundColor: 'rgba(110, 139, 87, 0.08)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  haloInner: {
    width: 110,
    height: 110,
    borderRadius: 55,
    backgroundColor: 'rgba(110, 139, 87, 0.16)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  recordButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: COLORS.primary,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 20,
    elevation: 10,
  },
  recordButtonActive: {
    backgroundColor: COLORS.danger,
    shadowColor: COLORS.danger,
  },
  statusText: {
    color: COLORS.textMuted,
    fontSize: 14,
    fontWeight: '600',
    marginTop: 22,
    marginBottom: 8,
  },
  recordPrompt: {
    color: COLORS.text,
    fontSize: 14.5,
    fontWeight: '600',
    textAlign: 'center',
    marginBottom: 20,
  },
  waveformWrapper: {
    height: 40,
    width: '100%',
    alignItems: 'center',
    justifyContent: 'center',
  },
  waveformRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    height: 40,
    width: 200,
  },
  card: {
    backgroundColor: COLORS.surface,
    padding: 18,
    borderRadius: 18,
    width: '100%',
    borderColor: COLORS.cardBorder,
    borderWidth: 1,
  },
  iconPrompt: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  iconPromptText: {
    color: COLORS.text,
    fontSize: 14.5,
    fontWeight: '600',
    marginLeft: 10,
    flex: 1,
  },
  lowConfidenceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  lowConfidenceNote: { color: COLORS.warning, fontSize: 12, marginLeft: 6 },
  cardBody: { color: COLORS.text, fontSize: 15, lineHeight: 22 },
  scoreWrap: {
    alignItems: 'center',
    width: '100%',
  },
  emotionBadge: {
    paddingHorizontal: 14,
    paddingVertical: 7,
    borderRadius: 20,
    borderWidth: 1,
    marginTop: 14,
    alignSelf: 'flex-start',
  },
  emotionBadgeText: { fontSize: 12.5, fontWeight: '600' },
  explanationText: { color: COLORS.textMuted, fontSize: 13.5, textAlign: 'center', marginTop: 18, marginBottom: 20, lineHeight: 19, paddingHorizontal: 6 },
  submitButton: {
    backgroundColor: COLORS.primary,
    paddingVertical: 13,
    paddingHorizontal: 20,
    borderRadius: 26,
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 16,
  },
  secondaryButton: {
    backgroundColor: 'transparent',
    borderWidth: 1.5,
    borderColor: COLORS.primary,
    marginTop: 10,
  },
  secondaryButtonText: {
    color: COLORS.primary,
  },
  skipText: {
    color: COLORS.textMuted,
    textAlign: 'center',
    marginTop: 14,
    fontSize: 13,
    fontWeight: '600',
  },
  photoPreview: {
    width: '100%',
    height: 200,
    borderRadius: 14,
    marginBottom: 14,
  },
  submitButtonText: {
    color: COLORS.onPrimary,
    fontWeight: '700',
    fontSize: 14.5,
  },
  submitButtonDisabled: {
    backgroundColor: COLORS.surfaceElevated,
    opacity: 0.6,
  },
  tagContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  soundTag: {
    backgroundColor: COLORS.surfaceElevated,
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
  },
  soundTagText: {
    color: COLORS.textMuted,
    fontSize: 13,
    fontWeight: '500',
  },
  dominantTag: {
    backgroundColor: 'rgba(110, 139, 87, 0.16)',
    borderColor: COLORS.primary,
    borderWidth: 1.5,
  },
  dominantTagText: {
    color: COLORS.primary,
    fontWeight: '700',
  },
});
