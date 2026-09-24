import React, { useState, useCallback, useEffect } from 'react';
import { View, Text, StyleSheet, SafeAreaView, ActivityIndicator, TouchableOpacity, Dimensions } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import Animated, { useSharedValue, useAnimatedStyle, withTiming, runOnJS, Extrapolation, interpolate } from 'react-native-reanimated';

import { Feather } from '@expo/vector-icons';
import { COLORS, getScoreMood } from '../theme/colors';
import { getDiaryEntries, EMOTION_DISPLAY } from '../utils/diaryStore';
import SkylineBanner from '../components/SkylineBanner';

const { width, height } = Dimensions.get('window');
const API_BASE = "http://127.0.0.1:5001";

const NOTE_SIZE = Math.min(width - 56, height * 0.5);

// Solid pastel paper colours per mood, with their own text pair: a sticky
// note stays paper-coloured regardless of the app theme.
const NOTE_PALETTE = {
  Calm: { paper: '#CFF3D9', text: '#1E3A2A', fold: '#B7DFC2' },
  Content: { paper: '#CFE6FF', text: '#193A5E', fold: '#B4D3EE' },
  Anxious: { paper: '#FFEAB0', text: '#5C4300', fold: '#F0D690' },
  Stressed: { paper: '#FFC9C9', text: '#5C1414', fold: '#F0AFAF' },
  Annoyed: { paper: '#FFD3EC', text: '#5C1440', fold: '#F0B9D8' },
};
const DEFAULT_PAPER = { paper: '#E7E5E0', text: '#3A3A38', fold: '#D3D0C9' };
const DEFAULT_DISPLAY = { icon: 'help-circle', color: COLORS.textMuted };

function dayName(timestamp) {
  const date = new Date(timestamp);
  const now = new Date();
  if (date.toDateString() === now.toDateString()) return 'Today';
  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  if (date.toDateString() === yesterday.toDateString()) return 'Yesterday';
  return date.toLocaleDateString([], { weekday: 'long' });
}

function fullDate(timestamp) {
  const date = new Date(timestamp);
  const time = date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  return `${date.toLocaleDateString([], { month: 'short', day: 'numeric' })} · ${time}`;
}

export default function DiaryScreen({ navigation }) {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [index, setIndex] = useState(0); // 0 = most recent entry

  // Reload on focus so an entry saved from RecordScreen appears immediately.
  useFocusEffect(
    useCallback(() => {
      let isActive = true;
      (async () => {
        setLoading(true);
        const loaded = await getDiaryEntries();
        if (isActive) {
          setEntries(loaded);
          setIndex(0);
          setLoading(false);
        }
      })();
      return () => { isActive = false; };
    }, [])
  );

  const [insight, setInsight] = useState('Save a few entries to see a pattern here.');
  useEffect(() => {
    const scored = entries.filter((e) => typeof e.wellbeingScore === 'number');
    if (scored.length === 0) return;
    let isActive = true;
    fetch(`${API_BASE}/api/diary_insight`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify({ entries: scored }),
    })
      .then(res => res.json())
      .then(data => { if (isActive && data.insight) setInsight(data.insight); })
      .catch(err => console.error('Failed to fetch diary insight:', err));
    return () => { isActive = false; };
  }, [entries]);

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.bannerBleed}>
        <SkylineBanner />
      </View>
      <View style={styles.header}>
        <Text style={styles.mainHeader}>Diary</Text>
        {entries.length > 0 && <Text style={styles.entryCount}>{entries.length} {entries.length === 1 ? 'entry' : 'entries'}</Text>}
      </View>

      <View style={styles.centerGroup}>
        {loading ? (
          <ActivityIndicator size="large" color={COLORS.textMuted} />
        ) : entries.length === 0 ? (
          <TouchableOpacity
            style={styles.emptyNote}
            activeOpacity={0.85}
            onPress={() => navigation?.navigate('Analyse')}
          >
            <Feather name="book-open" size={28} color={COLORS.textMuted} />
            <Text style={styles.emptyText}>No entries yet</Text>
            <View style={styles.emptyCta}>
              <Text style={styles.emptyCtaText}>Log your first moment</Text>
              <Feather name="arrow-right" size={14} color={COLORS.secondary} />
            </View>
          </TouchableOpacity>
        ) : (
          <FlipNotePad entries={entries} index={index} setIndex={setIndex} />
        )}

        {!loading && (
          <View style={styles.insightBar}>
            <Feather name="zap" size={12} color={COLORS.warning} />
            <Text style={styles.insightText}>{insight}</Text>
          </View>
        )}
      </View>
    </SafeAreaView>
  );
}

// One sticky note showing the current entry, with tap-to-flip paging.
// The flip is a two-phase rotateY (swapping content at edge-on) so it
// reads as turning a page.
function FlipNotePad({ entries, index, setIndex }) {
  const rotateY = useSharedValue(0);
  const [busy, setBusy] = useState(false);

  const flip = (delta) => {
    const next = index + delta;
    if (busy || next < 0 || next >= entries.length) {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
      return;
    }
    setBusy(true);
    Haptics.selectionAsync();
    rotateY.value = withTiming(90, { duration: 200 }, (finished) => {
      if (finished) {
        runOnJS(setIndex)(next);
        rotateY.value = withTiming(0, { duration: 200 }, () => {
          runOnJS(setBusy)(false);
        });
      }
    });
  };

  const animatedStyle = useAnimatedStyle(() => {
    const scale = interpolate(rotateY.value, [0, 90], [1, 0.94], Extrapolation.CLAMP);
    return {
      transform: [
        { perspective: 1400 },
        { rotateY: `${rotateY.value}deg` },
        { scale },
      ],
    };
  });

  const entry = entries[index];
  const display = (entry.emotion && EMOTION_DISPLAY[entry.emotion.label]) || DEFAULT_DISPLAY;
  const paper = (entry.emotion && NOTE_PALETTE[entry.emotion.label]) || DEFAULT_PAPER;
  const hasScore = typeof entry.wellbeingScore === 'number';
  const scorePct = hasScore ? Math.round(entry.wellbeingScore * 100) : null;

  return (
    <View style={styles.padArea}>
      <View style={styles.noteWrap}>
        <View style={[styles.notePin, { backgroundColor: display.color }]} />

        <Animated.View style={[styles.note, { backgroundColor: paper.paper }, animatedStyle]}>
          <View style={styles.noteTop}>
            <Text style={[styles.noteDay, { color: paper.text }]}>{dayName(entry.timestamp)}</Text>
            <Text style={[styles.noteDate, { color: `${paper.text}99` }]}>{fullDate(entry.timestamp)}</Text>
          </View>

          {hasScore ? (
            <View style={styles.noteScoreBlock}>
              <Text style={[styles.noteScoreValue, { color: paper.text }]}>{scorePct}</Text>
              <Text style={[styles.noteScoreMood, { color: `${paper.text}CC` }]}>{getScoreMood(entry.wellbeingScore)}</Text>
            </View>
          ) : (
            <Text style={[styles.noteMuted, { color: `${paper.text}99` }]}>No score</Text>
          )}

          <View style={styles.noteLabels}>
            {entry.emotion && (
              <View style={styles.noteLabelRow}>
                <Feather name={display.icon} size={13} color={paper.text} style={{ marginRight: 7 }} />
                <Text style={[styles.noteLabelText, { color: paper.text }]} numberOfLines={1}>{entry.emotion.label}</Text>
              </View>
            )}
            <View style={styles.noteLabelRow}>
              <Feather name="volume-2" size={13} color={paper.text} style={{ marginRight: 7 }} />
              <Text style={[styles.noteLabelText, { color: paper.text }]} numberOfLines={1}>{entry.soundLabel || 'Unknown sound'}</Text>
            </View>
            {entry.sceneLabel && (
              <View style={styles.noteLabelRow}>
                <Feather name="map-pin" size={13} color={paper.text} style={{ marginRight: 7 }} />
                <Text style={[styles.noteLabelText, { color: paper.text }]} numberOfLines={1}>{entry.sceneLabel}</Text>
              </View>
            )}
          </View>

          <View style={[styles.noteFold, { backgroundColor: paper.fold }]} />
        </Animated.View>
      </View>

      <View style={styles.pageControls}>
        <TouchableOpacity style={styles.pageBtn} onPress={() => flip(-1)} disabled={index === 0}>
          <Feather name="chevron-left" size={20} color={index === 0 ? COLORS.cardBorder : COLORS.text} />
        </TouchableOpacity>
        <Text style={styles.pageCount}>{index + 1} / {entries.length}</Text>
        <TouchableOpacity style={styles.pageBtn} onPress={() => flip(1)} disabled={index === entries.length - 1}>
          <Feather name="chevron-right" size={20} color={index === entries.length - 1 ? COLORS.cardBorder : COLORS.text} />
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.background },
  bannerBleed: { marginBottom: -6 },
  header: {
    flexDirection: 'row',
    alignItems: 'baseline',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    marginTop: 8,
  },
  mainHeader: { color: COLORS.text, fontSize: 20, fontWeight: '800' },
  entryCount: { color: COLORS.textMuted, fontSize: 12.5, fontWeight: '600' },

  // Centred as one group, so the insight sits under the paging controls.
  centerGroup: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  padArea: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  noteWrap: {
    width: NOTE_SIZE,
    height: NOTE_SIZE,
  },
  notePin: {
    position: 'absolute',
    top: -7,
    left: '50%',
    marginLeft: -6,
    width: 12,
    height: 12,
    borderRadius: 6,
    zIndex: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.4,
    shadowRadius: 2,
  },
  note: {
    flex: 1,
    borderRadius: 4,
    padding: 22,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.45,
    shadowRadius: 14,
    elevation: 10,
  },
  noteTop: {
    marginBottom: 18,
  },
  noteDay: { fontSize: 20, fontWeight: '800' },
  noteDate: { fontSize: 12.5, fontWeight: '600', marginTop: 2 },

  noteScoreBlock: {
    alignItems: 'flex-start',
    marginBottom: 20,
  },
  noteScoreValue: { fontSize: 48, fontWeight: '800', lineHeight: 52 },
  noteScoreMood: { fontSize: 13, fontWeight: '600', marginTop: 2 },
  noteMuted: { fontSize: 13, fontStyle: 'italic', marginBottom: 20 },

  noteLabels: {
    marginTop: 'auto',
  },
  noteLabelRow: { flexDirection: 'row', alignItems: 'center', marginTop: 6 },
  noteLabelText: { fontSize: 14, fontWeight: '600', flexShrink: 1 },

  noteFold: {
    position: 'absolute',
    bottom: -13,
    right: -13,
    width: 28,
    height: 28,
    transform: [{ rotate: '45deg' }],
  },

  pageControls: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 22,
  },
  pageBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: COLORS.surface,
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
    alignItems: 'center',
    justifyContent: 'center',
  },
  pageCount: {
    color: COLORS.textMuted,
    fontSize: 13,
    fontWeight: '600',
    marginHorizontal: 18,
  },

  insightBar: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: COLORS.surface,
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
    borderRadius: 14,
    paddingHorizontal: 16,
    paddingVertical: 12,
    marginTop: 26,
    width: NOTE_SIZE,
  },
  insightText: {
    color: COLORS.textMuted,
    fontSize: 12,
    marginLeft: 8,
    flex: 1,
    lineHeight: 16,
  },

  emptyNote: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 40,
  },
  emptyText: { color: COLORS.textMuted, fontSize: 14, fontWeight: '600', marginTop: 14 },
  emptyCta: { flexDirection: 'row', alignItems: 'center', marginTop: 10 },
  emptyCtaText: { color: COLORS.secondary, fontSize: 13, fontWeight: '600', marginRight: 6 },
});
