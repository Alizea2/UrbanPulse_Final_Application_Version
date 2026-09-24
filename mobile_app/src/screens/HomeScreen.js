import React, { useState, useCallback, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Dimensions, SafeAreaView } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { Feather } from '@expo/vector-icons';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withSequence,
  withDelay,
  withTiming,
} from 'react-native-reanimated';
import * as Haptics from 'expo-haptics';
import { COLORS, getScoreColor } from '../theme/colors';
import { getDiaryEntries, EMOTION_DISPLAY } from '../utils/diaryStore';
import { getMapPins } from '../utils/mapStore';
import ScoreRing from '../components/ScoreRing';
import SkylineBanner from '../components/SkylineBanner';

const { width } = Dimensions.get('window');

const QUIET_SPOT_THRESHOLD = 0.7;          // score a pin needs to count as quiet
const WEEK_MS = 7 * 24 * 60 * 60 * 1000;   // window for the weekly average

// Turns a timestamp into "Just now" / "5m ago" / "2d ago".
function relativeTime(ts) {
  const diffMin = Math.floor((Date.now() - ts) / 60000);
  if (diffMin < 1) return 'Just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  const hr = Math.floor(diffMin / 60);
  if (hr < 24) return `${hr}h ago`;
  return `${Math.floor(hr / 24)}d ago`;
}

// The pulsing record button: the one primary action on this screen.
function RecordButton({ onPress }) {
  const pulse = useSharedValue(1);

  useEffect(() => {
    pulse.value = withDelay(
      400,
      withRepeat(withSequence(withTiming(1.08, { duration: 900 }), withTiming(1, { duration: 900 })), -1, true)
    );
  }, []);

  const animatedStyle = useAnimatedStyle(() => ({ transform: [{ scale: pulse.value }] }));

  return (
    <TouchableOpacity activeOpacity={0.85} onPress={onPress}>
      <Animated.View style={[styles.recordButton, animatedStyle]}>
        <Feather name="mic" size={26} color={COLORS.onPrimary} />
        <Text style={styles.recordButtonText}>Record</Text>
      </Animated.View>
    </TouchableOpacity>
  );
}

function StatTile({ icon, value, label, color }) {
  return (
    <View style={styles.statTile}>
      <Feather name={icon} size={16} color={color || COLORS.secondary} />
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

export default function HomeScreen({ navigation }) {
  const [entries, setEntries] = useState([]);
  const [pins, setPins] = useState([]);
  const [ready, setReady] = useState(false);

  // Reloads on focus, so an entry saved on another tab shows up here.
  useFocusEffect(
    useCallback(() => {
      // Guards against setting state after the screen has gone away.
      let active = true;
      (async () => {
        // Fetched together, since neither depends on the other.
        const [entryList, pinList] = await Promise.all([getDiaryEntries(), getMapPins()]);
        if (!active) return;
        setEntries(entryList);
        setPins(pinList);
        setReady(true);
      })();
      return () => { active = false; };
    }, [])
  );

  // Older entries predate scoring, so only scored ones feed the stats.
  const scored = entries.filter((e) => typeof e.wellbeingScore === 'number');
  const hasData = scored.length > 0;
  const latestScore = hasData ? scored[0].wellbeingScore : 0.5;

  // Weekly average across this user's own entries.
  const weekAgo = Date.now() - WEEK_MS;
  const weekScored = scored.filter((e) => e.timestamp >= weekAgo);
  const weekAvg = weekScored.length
    ? (weekScored.reduce((sum, e) => sum + e.wellbeingScore, 0) / weekScored.length).toFixed(2)
    : '—';

  // Counted from the shared map, not just this user's entries.
  const quietSpotCount = pins.filter((p) => typeof p.wellbeingScore === 'number' && p.wellbeingScore >= QUIET_SPOT_THRESHOLD).length;

  const recent = entries.slice(0, 4);

  return (
    <SafeAreaView style={styles.container}>
    <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      <View style={styles.bannerBleed}>
        <SkylineBanner />
      </View>

      <View style={styles.header}>
        <Text style={styles.wordmark}>UrbanPulse</Text>
        <Text style={styles.tagline}>Map your city's soundscape, find calm, avoid noise</Text>
      </View>

      <View style={styles.ringWrap}>
        <ScoreRing score={latestScore} hasData={hasData} />
      </View>

      <RecordButton onPress={() => navigation.navigate('Analyse')} />

      {hasData && (
        <View style={styles.statsRow}>
          <StatTile icon="list" value={entries.length} label="Entries" color={COLORS.secondary} />
          <StatTile icon="trending-up" value={weekAvg} label="7-day avg" color={getScoreColor(hasData ? parseFloat(weekAvg) || latestScore : 0.5)} />
          <StatTile icon="map-pin" value={quietSpotCount} label="Quiet spots" color={COLORS.primary} />
        </View>
      )}

      <View style={styles.section}>
        <Text style={styles.sectionLabel}>RECENT</Text>

        {!ready ? null : !hasData ? (
          <TouchableOpacity
            style={styles.emptyCard}
            activeOpacity={0.85}
            onPress={() => navigation.navigate('Analyse')}
          >
            <Feather name="activity" size={22} color={COLORS.textMuted} />
            <Text style={styles.emptyCardText}>Log your first moment to see your Pulse build here.</Text>
            <Feather name="arrow-right" size={16} color={COLORS.secondary} />
          </TouchableOpacity>
        ) : (
          recent.map((entry) => {
            const emotionMeta = EMOTION_DISPLAY[entry.emotion?.label] || { icon: 'circle', color: COLORS.textMuted };
            const dotColor = typeof entry.wellbeingScore === 'number' ? getScoreColor(entry.wellbeingScore) : COLORS.textMuted;
            return (
              <TouchableOpacity
                key={entry.id}
                style={styles.activityRow}
                activeOpacity={0.7}
                onPress={() => navigation.navigate('Diary')}
              >
                <View style={[styles.activityDot, { backgroundColor: dotColor }]} />
                <Feather name={emotionMeta.icon} size={16} color={emotionMeta.color} style={{ marginRight: 10 }} />
                <Text style={styles.activityLabel} numberOfLines={1}>{entry.soundLabel || 'Unknown sound'}</Text>
                <Text style={styles.activityTime}>{relativeTime(entry.timestamp)}</Text>
              </TouchableOpacity>
            );
          })
        )}
      </View>
    </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  content: {
    paddingHorizontal: 24,
    paddingTop: 0,
    paddingBottom: 40,
    alignItems: 'center',
  },
  bannerBleed: {
    // The shared SkylineBanner is full device-width — this cancels out
    // the screen's own horizontal padding so it bleeds to the edges.
    marginHorizontal: -24,
  },
  header: {
    width: '100%',
    marginTop: -14,
    marginBottom: 26,
    alignItems: 'center',
  },
  wordmark: {
    color: COLORS.text,
    fontSize: 34,
    fontWeight: '800',
    textAlign: 'center',
  },
  tagline: {
    color: COLORS.textMuted,
    fontSize: 13.5,
    fontWeight: '600',
    marginTop: 6,
    textAlign: 'center',
  },
  ringWrap: {
    marginBottom: 22,
  },
  recordButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.primary,
    paddingVertical: 14,
    paddingHorizontal: 30,
    borderRadius: 30,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 16,
    elevation: 8,
  },
  recordButtonText: {
    color: COLORS.onPrimary,
    fontWeight: '800',
    fontSize: 15,
    marginLeft: 8,
  },
  statsRow: {
    flexDirection: 'row',
    width: '100%',
    marginTop: 28,
    gap: 10,
  },
  statTile: {
    flex: 1,
    backgroundColor: COLORS.surface,
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
    borderRadius: 14,
    paddingVertical: 14,
    alignItems: 'center',
  },
  statValue: {
    color: COLORS.text,
    fontSize: 17,
    fontWeight: '700',
    marginTop: 6,
  },
  statLabel: {
    color: COLORS.textMuted,
    fontSize: 10.5,
    marginTop: 2,
  },
  section: {
    width: '100%',
    marginTop: 32,
  },
  sectionLabel: {
    color: COLORS.textMuted,
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 1.2,
    marginBottom: 12,
  },
  activityRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.surface,
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 14,
    marginBottom: 10,
  },
  activityDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 10,
  },
  activityLabel: {
    flex: 1,
    color: COLORS.text,
    fontSize: 13.5,
    fontWeight: '600',
  },
  activityTime: {
    color: COLORS.textMuted,
    fontSize: 11.5,
  },
  emptyCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.surface,
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
    borderRadius: 14,
    padding: 16,
  },
  emptyCardText: {
    flex: 1,
    color: COLORS.textMuted,
    fontSize: 12.5,
    marginHorizontal: 12,
    lineHeight: 17,
  },
});
