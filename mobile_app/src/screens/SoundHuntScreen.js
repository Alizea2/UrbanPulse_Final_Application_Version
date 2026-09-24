import React, { useState, useCallback } from 'react';
import { View, Text, StyleSheet, SafeAreaView, ScrollView, TouchableOpacity, ActivityIndicator, Alert } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import { useAudioRecorder, AudioModule, setAudioModeAsync, IOSOutputFormat, AudioQuality } from 'expo-audio';
import * as FileSystem from 'expo-file-system/legacy';
import Animated, { useSharedValue, useAnimatedStyle, withRepeat, withSequence, withTiming } from 'react-native-reanimated';

import { Feather } from '@expo/vector-icons';
import { COLORS } from '../theme/colors';
import { getCompletedChallenges, completeChallenge } from '../utils/soundHuntStore';
import SkylineBanner from '../components/SkylineBanner';

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
  android: { extension: '.wav', outputFormat: 'default', audioEncoder: 'default' },
  web: { mimeType: 'audio/webm' },
};

// A challenge is found if any of the recording's top-3 labels contains one
// of its keywords. Three kinds: keywords only, time only (keywords: null),
// or both. `time` is {start, end} in 24h local time; end < start wraps past
// midnight (e.g. Night Owl, 22 -> 2).
const CHALLENGES = [
  // sound-only
  { id: 'rain', title: 'Urban Rain', target: 'rainfall', points: 30, icon: 'cloud-rain', color: '#3b82f6', keywords: ['rain'], time: null },
  { id: 'silence', title: 'Absolute Silence', target: 'near-silence', points: 100, icon: 'mic-off', color: '#8b5cf6', keywords: ['silence'], time: null },
  { id: 'cafe', title: 'Cafe Ambience', target: 'café chatter', points: 40, icon: 'coffee', color: '#f59e0b', keywords: ['speech', 'chatter', 'dish', 'cup', 'clink'], time: null },
  { id: 'traffic', title: 'City Traffic', target: 'traffic noise', points: 20, icon: 'truck', color: '#ef4444', keywords: ['traffic', 'vehicle', 'engine', 'car'], time: null },
  { id: 'music', title: 'Music in the Air', target: 'music', points: 25, icon: 'music', color: '#ec4899', keywords: ['music'], time: null },
  { id: 'dog', title: 'Good Boy', target: 'a dog', points: 25, icon: 'github', color: '#a3e635', keywords: ['dog', 'bark'], time: null },

  // time-only — any successful recording counts, but only in the window
  { id: 'early_riser', title: 'Early Riser', target: 'anything, before 8am', points: 35, icon: 'sunrise', color: '#fb923c', keywords: null, time: { start: 5, end: 8 } },
  { id: 'night_owl', title: 'Night Owl', target: 'anything, after 10pm', points: 35, icon: 'moon', color: '#818cf8', keywords: null, time: { start: 22, end: 2 } },

  // both — sound AND time window
  { id: 'dawn_chorus', title: 'Dawn Chorus', target: 'bird calls, 5am–11am', points: 60, icon: 'feather', color: '#10b981', keywords: ['bird'], time: { start: 5, end: 11 } },
  { id: 'midnight_quiet', title: 'Midnight Quiet', target: 'silence, 10pm–4am', points: 120, icon: 'moon', color: '#6366f1', keywords: ['silence'], time: { start: 22, end: 4 } },
  { id: 'coffee_rush', title: 'Coffee Hour Rush', target: 'café chatter, 7am–11am', points: 55, icon: 'coffee', color: '#d97706', keywords: ['speech', 'chatter', 'dish', 'cup', 'clink'], time: { start: 7, end: 11 } },
  { id: 'rush_hour', title: 'Rush Hour Roar', target: 'traffic, 5pm–7pm', points: 45, icon: 'truck', color: '#dc2626', keywords: ['traffic', 'vehicle', 'engine', 'car'], time: { start: 17, end: 19 } },
];

// Word-boundary matching, not substring: .includes() let "rain" match
// "Train" and "car" match "scared".
function matchesSound(results, keywords) {
  if (!keywords) return true; // time-only challenge — any recording counts
  return (results || []).some(r => {
    const label = (r.label || '').toLowerCase();
    return keywords.some(k => new RegExp(`\\b${k}\\b`).test(label));
  });
}

function isWithinTimeWindow(time, now = new Date()) {
  if (!time) return true;
  const hour = now.getHours();
  if (time.start <= time.end) return hour >= time.start && hour < time.end;
  return hour >= time.start || hour < time.end; // wraps past midnight
}

function formatHour(h) {
  const period = h >= 12 ? 'pm' : 'am';
  const display = h % 12 === 0 ? 12 : h % 12;
  return `${display}${period}`;
}

function formatTimeWindow(time) {
  if (!time) return null;
  return `${formatHour(time.start)}–${formatHour(time.end)}`;
}

export default function SoundHuntScreen() {
  const [completedIds, setCompletedIds] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeId, setActiveId] = useState(null);
  const [phase, setPhase] = useState('idle'); // idle, recording, checking, miss
  const audioRecorder = useAudioRecorder(WAV_RECORDING_OPTIONS);

  useFocusEffect(
    useCallback(() => {
      let isActive = true;
      (async () => {
        setLoading(true);
        const ids = await getCompletedChallenges();
        if (isActive) {
          setCompletedIds(ids);
          setLoading(false);
        }
      })();
      return () => { isActive = false; };
    }, [])
  );

  const completed = CHALLENGES.filter(c => completedIds.includes(c.id));
  const available = CHALLENGES.filter(c => !completedIds.includes(c.id));
  const totalXp = completed.reduce((sum, c) => sum + c.points, 0);
  const active = CHALLENGES.find(c => c.id === activeId);
  const inWindow = active ? isWithinTimeWindow(active.time) : true;

  const startHunt = (id) => {
    setActiveId(id);
    setPhase('idle');
  };

  const cancelHunt = () => {
    setActiveId(null);
    setPhase('idle');
  };

  const handleRecordPress = async () => {
    if ((phase === 'idle' || phase === 'miss') && !isWithinTimeWindow(active.time)) {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
      return; // out-of-window taps are a no-op — the status text already explains why
    }
    try {
      if (phase === 'idle' || phase === 'miss') {
        const perm = await AudioModule.requestRecordingPermissionsAsync();
        if (perm.status !== 'granted') throw new Error('Microphone permission not granted');
        await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });
        await audioRecorder.prepareToRecordAsync(WAV_RECORDING_OPTIONS);
        audioRecorder.record();
        setPhase('recording');
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
      } else if (phase === 'recording') {
        setPhase('checking');
        try {
          await audioRecorder.stop();
        } catch (e) { /* already stopped */ }
        await new Promise(resolve => setTimeout(resolve, 400));

        let fileUri = audioRecorder.uri;
        if (!fileUri) throw new Error('No recording captured');
        if (!fileUri.startsWith('file://')) fileUri = 'file://' + fileUri;

        const base64Audio = await FileSystem.readAsStringAsync(fileUri, { encoding: FileSystem.EncodingType.Base64 });
        const response = await fetch(`${API_BASE}/api/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
          body: JSON.stringify({ audio: base64Audio }),
        });
        if (!response.ok) throw new Error(`Server returned ${response.status}`);
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        if (matchesSound(data.results, active.keywords) && isWithinTimeWindow(active.time)) {
          // The server re-verifies this recording itself, so this call can
          // still fail (e.g. if the device clock disagrees on the window).
          await completeChallenge(active.id, base64Audio);
          Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
          setCompletedIds(prev => [...new Set([...prev, active.id])]);
          setActiveId(null);
          setPhase('idle');
        } else {
          Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
          setPhase('miss');
        }
      }
    } catch (err) {
      console.error('Sound Hunt error:', err);
      Alert.alert('Hunt Failed', err.message || 'Something went wrong.');
      setPhase('idle');
    }
  };

  const pulse = useSharedValue(1);
  React.useEffect(() => {
    if (phase === 'recording') {
      pulse.value = withRepeat(withSequence(withTiming(1.12, { duration: 500 }), withTiming(1, { duration: 500 })), -1, true);
    } else {
      pulse.value = withTiming(1, { duration: 200 });
    }
  }, [phase]);
  const pulseStyle = useAnimatedStyle(() => ({ transform: [{ scale: pulse.value }] }));

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scroll}>

        <View style={styles.bannerBleed}>
          <SkylineBanner />
        </View>

        <View style={styles.headerRow}>
          <Text style={styles.headerTitle}>Sound Hunt</Text>
          <Text style={styles.headerXp}>{totalXp} XP</Text>
        </View>
        <Text style={styles.headerTagline}>Complete missions by recording real sounds to earn badges</Text>

        {/* Badges */}
        <View style={styles.section}>
          <Text style={styles.sectionLabel}>BADGES</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            {loading ? (
              <ActivityIndicator size="small" color={COLORS.textMuted} />
            ) : completed.length === 0 ? (
              <Text style={styles.emptyText}>Complete a hunt to earn your first badge</Text>
            ) : (
              completed.map(c => (
                <View key={c.id} style={styles.badgeContainer}>
                  <View style={[styles.badgeOrb, { borderColor: c.color, backgroundColor: `${c.color}20` }]}>
                    <Feather name={c.icon} size={24} color={c.color} />
                  </View>
                  <Text style={styles.badgeTitle} numberOfLines={1}>{c.title}</Text>
                </View>
              ))
            )}
          </ScrollView>
        </View>

        {/* Active hunt */}
        {active && (
          <View style={styles.section}>
            <Text style={styles.sectionLabel}>CURRENT HUNT</Text>
            <View style={[styles.activeCard, { borderColor: `${active.color}55` }]}>
              <TouchableOpacity style={styles.activeClose} onPress={cancelHunt}>
                <Feather name="x" size={16} color={COLORS.textMuted} />
              </TouchableOpacity>

              <View style={[styles.activeIcon, { backgroundColor: `${active.color}25` }]}>
                <Feather name={active.icon} size={26} color={active.color} />
              </View>
              <Text style={styles.activeTitle}>{active.title}</Text>
              <Text style={styles.activeTarget}>Listening for {active.target}</Text>

              {active.time && (
                <View style={styles.windowBadge}>
                  <Feather name="clock" size={11} color={inWindow ? COLORS.primary : COLORS.textMuted} />
                  <Text style={[styles.windowBadgeText, { color: inWindow ? COLORS.primary : COLORS.textMuted }]}>
                    {formatTimeWindow(active.time)}
                  </Text>
                </View>
              )}

              <TouchableOpacity
                onPress={handleRecordPress}
                disabled={phase === 'checking' || !inWindow}
                style={[
                  styles.recordBtn,
                  { backgroundColor: phase === 'recording' ? COLORS.danger : active.color },
                  !inWindow && styles.recordBtnDisabled,
                ]}
              >
                <Animated.View style={pulseStyle}>
                  {phase === 'checking' ? (
                    <ActivityIndicator size="small" color={COLORS.onPrimary} />
                  ) : (
                    <Feather name={phase === 'recording' ? 'square' : 'mic'} size={22} color={COLORS.onPrimary} />
                  )}
                </Animated.View>
              </TouchableOpacity>

              <Text style={styles.activeStatus}>
                {!inWindow && `Only available ${formatTimeWindow(active.time)}`}
                {inWindow && phase === 'idle' && 'Tap to record'}
                {inWindow && phase === 'recording' && 'Recording — tap to stop'}
                {inWindow && phase === 'checking' && 'Checking…'}
                {inWindow && phase === 'miss' && "Didn't match — try again"}
              </Text>
            </View>
          </View>
        )}

        {/* Available */}
        <View style={styles.section}>
          <Text style={styles.sectionLabel}>MISSIONS</Text>
          {available.filter(c => c.id !== activeId).map(c => (
            <TouchableOpacity key={c.id} style={styles.missionCard} onPress={() => startHunt(c.id)}>
              <View style={[styles.missionIcon, { backgroundColor: `${c.color}18` }]}>
                <Feather name={c.icon} size={20} color={c.color} />
              </View>
              <View style={styles.missionInfo}>
                <Text style={styles.missionTitle}>{c.title}</Text>
                <Text style={styles.missionTarget}>Find {c.target}</Text>
              </View>
              <Text style={[styles.missionPoints, { color: c.color }]}>+{c.points}</Text>
            </TouchableOpacity>
          ))}
          {available.length === 0 && !loading && (
            <Text style={styles.emptyText}>All hunts complete — nice work</Text>
          )}
        </View>

        <View style={{ height: 30 }} />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.background },
  scroll: { paddingHorizontal: 20, paddingTop: 0 },
  bannerBleed: { marginHorizontal: -20, marginBottom: 10 },

  headerRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  headerTitle: { color: COLORS.text, fontSize: 20, fontWeight: '800' },
  headerXp: { color: COLORS.primary, fontSize: 13, fontWeight: '700' },
  headerTagline: { color: COLORS.textMuted, fontSize: 12.5, fontWeight: '600', marginTop: 4, marginBottom: 20 },

  section: { marginBottom: 26 },
  sectionLabel: { color: COLORS.secondary, fontSize: 11, fontWeight: '700', letterSpacing: 1.3, marginBottom: 12 },
  emptyText: { color: COLORS.textMuted, fontSize: 12.5, fontStyle: 'italic' },

  badgeContainer: { alignItems: 'center', marginRight: 18, width: 72 },
  badgeOrb: {
    width: 54, height: 54, borderRadius: 27, borderWidth: 1.5,
    alignItems: 'center', justifyContent: 'center', marginBottom: 6,
  },
  badgeTitle: { color: COLORS.textMuted, fontSize: 10.5, fontWeight: '700', textAlign: 'center' },

  activeCard: {
    backgroundColor: COLORS.surface,
    borderRadius: 20,
    padding: 22,
    borderWidth: 1,
    alignItems: 'center',
  },
  activeClose: { position: 'absolute', top: 12, right: 12, zIndex: 2 },
  activeIcon: { width: 52, height: 52, borderRadius: 26, alignItems: 'center', justifyContent: 'center', marginBottom: 12 },
  activeTitle: { color: COLORS.text, fontSize: 17, fontWeight: '800' },
  activeTarget: { color: COLORS.textMuted, fontSize: 12.5, fontWeight: '600', marginTop: 3 },
  windowBadge: {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: COLORS.surfaceElevated,
    paddingHorizontal: 10, paddingVertical: 4, borderRadius: 12,
    marginTop: 10, marginBottom: 8,
  },
  windowBadgeText: { fontSize: 11, fontWeight: '700', marginLeft: 5 },
  recordBtn: {
    width: 64, height: 64, borderRadius: 32,
    alignItems: 'center', justifyContent: 'center',
    marginTop: 10,
    shadowColor: '#000', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.3, shadowRadius: 8,
  },
  recordBtnDisabled: { opacity: 0.35 },
  activeStatus: { color: COLORS.textMuted, fontSize: 12.5, fontWeight: '600', marginTop: 14 },

  missionCard: {
    flexDirection: 'row',
    backgroundColor: COLORS.surface,
    borderRadius: 16,
    padding: 14,
    marginBottom: 10,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
  },
  missionIcon: { width: 42, height: 42, borderRadius: 21, alignItems: 'center', justifyContent: 'center', marginRight: 12 },
  missionInfo: { flex: 1 },
  missionTitle: { color: COLORS.text, fontSize: 14.5, fontWeight: '700' },
  missionTarget: { color: COLORS.textMuted, fontSize: 12, marginTop: 2 },
  missionPoints: { fontSize: 13, fontWeight: '800' },
});
