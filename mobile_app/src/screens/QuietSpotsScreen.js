import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { View, Text, StyleSheet, SafeAreaView, TouchableOpacity, Dimensions, ActivityIndicator } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import * as Location from 'expo-location';
import Animated, {
  FadeIn, FadeInDown, FadeOut,
  useSharedValue, useAnimatedStyle, withRepeat, withTiming, withDelay, Easing,
} from 'react-native-reanimated';

import { Feather } from '@expo/vector-icons';
import { COLORS, getScoreColor } from '../theme/colors';
import { getMapPins } from '../utils/mapStore';
import SkylineBanner from '../components/SkylineBanner';

const { width } = Dimensions.get('window');
const RADAR_SIZE = Math.min(width - 48, 340);
const RADAR_CENTER = RADAR_SIZE / 2;
const BLIP_MARGIN = 20; // keeps a blip's own circle from clipping the radar edge

// A quiet spot must be Calm AND meet this score: either condition alone
// can be met for unrelated reasons, so both have to hold.
const MIN_QUIET_SCORE = 0.50;
const MAX_RESULTS = 8;

// Selectable search radii (km). Without a bound, ranking by score alone
// could surface a pin on the other side of the world.
const RADIUS_OPTIONS = [15, 25, 40];
const DEFAULT_RADIUS_KM = 25;

function haversineDistanceKm(a, b) {
  const R = 6371;
  const dLat = ((b.latitude - a.latitude) * Math.PI) / 180;
  const dLon = ((b.longitude - a.longitude) * Math.PI) / 180;
  const lat1 = (a.latitude * Math.PI) / 180;
  const lat2 = (b.latitude * Math.PI) / 180;
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}

// True compass bearing (0 = North), so blips sit in their real direction.
function bearingDegrees(from, to) {
  const lat1 = (from.latitude * Math.PI) / 180;
  const lat2 = (to.latitude * Math.PI) / 180;
  const dLon = ((to.longitude - from.longitude) * Math.PI) / 180;
  const y = Math.sin(dLon) * Math.cos(lat2);
  const x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);
  return ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360;
}

function iconForSound(desc) {
  const d = (desc || '').toLowerCase();
  if (d.includes('bird')) return 'feather';
  if (d.includes('wind')) return 'wind';
  if (d.includes('water')) return 'droplet';
  if (d.includes('music')) return 'music';
  if (d.includes('speech') || d.includes('chatter') || d.includes('voice')) return 'users';
  return 'sun';
}

function formatDistance(km) {
  if (km < 1) return `${Math.round(km * 1000)}m away`;
  return `${km.toFixed(1)}km away`;
}

// One expanding, fading sonar ring; several staggered ones animate the radar.
function Ping({ delay }) {
  const progress = useSharedValue(0);
  useEffect(() => {
    progress.value = withDelay(delay, withRepeat(withTiming(1, { duration: 2600, easing: Easing.out(Easing.ease) }), -1, false));
  }, []);
  const style = useAnimatedStyle(() => ({
    opacity: (1 - progress.value) * 0.5,
    transform: [{ scale: 0.15 + progress.value * 0.9 }],
  }));
  return <Animated.View style={[styles.ping, style]} />;
}

// A quiet spot plotted at its real bearing + distance from the user.
function Blip({ spot, angle, radius, onPress, isTop }) {
  const rad = (angle * Math.PI) / 180;
  const x = RADAR_CENTER + radius * Math.sin(rad);
  const y = RADAR_CENTER - radius * Math.cos(rad);
  const color = getScoreColor(spot.wellbeingScore);

  const pulse = useSharedValue(1);
  useEffect(() => {
    if (!isTop) return;
    pulse.value = withRepeat(withTiming(1.25, { duration: 900, easing: Easing.inOut(Easing.ease) }), -1, true);
  }, [isTop]);
  const pulseStyle = useAnimatedStyle(() => ({ transform: [{ scale: isTop ? pulse.value : 1 }] }));

  return (
    <Animated.View entering={FadeIn.delay(200)} style={[styles.blipWrap, { left: x - 16, top: y - 16 }]}>
      {isTop && <Animated.View style={[styles.blipHalo, pulseStyle]} />}
      <TouchableOpacity onPress={() => onPress(spot)} style={[styles.blip, { backgroundColor: color, borderColor: color }]}>
        <Feather name={iconForSound(spot.desc)} size={13} color={COLORS.onPrimary} />
      </TouchableOpacity>
    </Animated.View>
  );
}

export default function QuietSpotsScreen({ navigation }) {
  const [pins, setPins] = useState([]);
  const [userLoc, setUserLoc] = useState(null);
  const [radiusKm, setRadiusKm] = useState(DEFAULT_RADIUS_KM);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState(null);
  const [selected, setSelected] = useState(null);

  useFocusEffect(
    useCallback(() => {
      let isActive = true;
      (async () => {
        setLoading(true);
        setErrorMsg(null);
        setSelected(null);
        try {
          const perm = await Location.requestForegroundPermissionsAsync();
          if (perm.status !== 'granted') {
            if (isActive) {
              setErrorMsg('Location access is needed to find quiet spots near you.');
              setLoading(false);
            }
            return;
          }
          // The Simulator throws a transient kCLErrorLocationUnknown; one
          // retry resolves most cases without showing the user an error.
          let position;
          try {
            position = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Highest });
          } catch (firstErr) {
            console.warn('Location fix failed once, retrying:', firstErr.message);
            await new Promise(resolve => setTimeout(resolve, 1200));
            position = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Highest });
          }
          const loc = { latitude: position.coords.latitude, longitude: position.coords.longitude };

          const fetchedPins = await getMapPins();
          if (isActive) {
            setUserLoc(loc);
            setPins(fetchedPins);
          }
        } catch (err) {
          // warn, not error: LogBox turns errors into a full-screen overlay.
          console.warn('Failed to load quiet spots:', err.message);
          if (isActive) setErrorMsg('Could not find your location.');
        } finally {
          if (isActive) setLoading(false);
        }
      })();
      return () => { isActive = false; };
    }, [])
  );

  // Recomputed locally on radius change; no re-fetch needed.
  const spots = useMemo(() => {
    if (!userLoc) return [];
    return pins
      .map(p => ({
        ...p,
        distanceKm: haversineDistanceKm(userLoc, p.loc),
        bearing: bearingDegrees(userLoc, p.loc),
      }))
      .filter(p => p.emotion === 'Calm' && typeof p.wellbeingScore === 'number' && p.wellbeingScore >= MIN_QUIET_SCORE && p.distanceKm <= radiusKm)
      .sort((a, b) => b.wellbeingScore - a.wellbeingScore)
      .slice(0, MAX_RESULTS);
  }, [pins, userLoc, radiusKm]);

  useEffect(() => {
    setSelected(null);
  }, [radiusKm]);

  const findOnMap = (spot) => {
    navigation?.navigate('Map', { focusLocation: spot.loc, focusPinId: spot.id });
  };

  // Ring position encodes score, not distance: the best spot sits at the
  // centre, tapering outward as score approaches MIN_QUIET_SCORE.
  const maxScore = spots.length > 0 ? Math.max(...spots.map(s => s.wellbeingScore), MIN_QUIET_SCORE + 0.01) : 1;
  const usableRadius = RADAR_CENTER - BLIP_MARGIN;

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.bannerBleed}>
        <SkylineBanner />
      </View>
      <View style={styles.headerContainer}>
        <Text style={styles.headerTitle}>Quiet Spot Finder</Text>
        {spots.length > 0 && <Text style={styles.headerDesc}>{spots.length} calm spot{spots.length === 1 ? '' : 's'} within {radiusKm}km — best at centre</Text>}

        <View style={styles.radiusRow}>
          {RADIUS_OPTIONS.map((km) => {
            const active = km === radiusKm;
            return (
              <TouchableOpacity
                key={km}
                onPress={() => setRadiusKm(km)}
                style={[styles.radiusPill, active && styles.radiusPillActive]}
              >
                <Text style={[styles.radiusPillText, active && styles.radiusPillTextActive]}>{km}km</Text>
              </TouchableOpacity>
            );
          })}
        </View>
      </View>

      {loading ? (
        <ActivityIndicator size="large" color={COLORS.textMuted} style={{ marginTop: 60 }} />
      ) : errorMsg ? (
        <View style={styles.emptyState}>
          <Feather name="map-pin" size={32} color={COLORS.textMuted} />
          <Text style={styles.emptyText}>{errorMsg}</Text>
        </View>
      ) : spots.length === 0 ? (
        <View style={styles.emptyState}>
          <Feather name="wind" size={32} color={COLORS.textMuted} />
          <Text style={styles.emptyText}>No calm spots within {radiusKm}km yet{'\n'}Try a wider radius above</Text>
        </View>
      ) : (
        <View style={styles.radarArea}>
          <View style={styles.radar}>
            <View style={[styles.ring, { width: RADAR_SIZE, height: RADAR_SIZE, borderRadius: RADAR_CENTER }]} />
            <View style={[styles.ring, { width: RADAR_SIZE * 0.66, height: RADAR_SIZE * 0.66, borderRadius: RADAR_CENTER * 0.66 }]} />
            <View style={[styles.ring, { width: RADAR_SIZE * 0.33, height: RADAR_SIZE * 0.33, borderRadius: RADAR_CENTER * 0.33 }]} />

            <Ping delay={0} />
            <Ping delay={900} />
            <Ping delay={1800} />

            <View style={styles.youDot}>
              <Feather name="navigation-2" size={12} color={COLORS.onPrimary} />
            </View>

            {spots.map((spot, i) => (
              <Blip
                key={spot.id}
                spot={spot}
                angle={spot.bearing}
                radius={Math.max(28, ((maxScore - spot.wellbeingScore) / (maxScore - MIN_QUIET_SCORE)) * usableRadius)}
                onPress={setSelected}
                isTop={i === 0}
              />
            ))}
          </View>

          {selected ? (
            <Animated.View entering={FadeInDown.duration(250)} exiting={FadeOut.duration(150)} style={styles.detailCard}>
              <TouchableOpacity style={styles.detailClose} onPress={() => setSelected(null)}>
                <Feather name="x" size={16} color={COLORS.textMuted} />
              </TouchableOpacity>
              <View style={styles.detailHeader}>
                <View style={[styles.detailIcon, { backgroundColor: getScoreColor(selected.wellbeingScore) }]}>
                  <Feather name={iconForSound(selected.desc)} size={18} color={COLORS.onPrimary} />
                </View>
                <View style={{ flex: 1, marginLeft: 12 }}>
                  <Text style={styles.detailTitle} numberOfLines={1}>{selected.desc || 'Quiet Spot'}</Text>
                  <Text style={styles.detailMeta}>{formatDistance(selected.distanceKm)}</Text>
                </View>
                <Text style={[styles.detailScore, { color: getScoreColor(selected.wellbeingScore) }]}>{Math.round(selected.wellbeingScore * 100)}</Text>
              </View>
              <TouchableOpacity
                style={[styles.mapBtn, { backgroundColor: getScoreColor(selected.wellbeingScore) }]}
                onPress={() => findOnMap(selected)}
              >
                <Feather name="map" size={16} color={COLORS.onPrimary} style={{ marginRight: 6 }} />
                <Text style={styles.mapBtnText}>Find on Map</Text>
              </TouchableOpacity>
            </Animated.View>
          ) : (
            <Text style={styles.hint}>Tap a colored ping to see the spot</Text>
          )}
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.background },

  bannerBleed: {
    marginBottom: -6,
  },
  headerContainer: {
    paddingHorizontal: 25,
    paddingTop: 6,
    paddingBottom: 8,
    alignItems: 'center',
  },
  headerTitle: { color: COLORS.text, fontSize: 20, fontWeight: '800' },
  headerDesc: { color: COLORS.textMuted, fontSize: 12.5, fontWeight: '600', marginTop: 4 },

  radiusRow: {
    flexDirection: 'row',
    marginTop: 12,
  },
  radiusPill: {
    paddingHorizontal: 16,
    paddingVertical: 7,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
    backgroundColor: COLORS.surface,
    marginHorizontal: 5,
  },
  radiusPillActive: {
    backgroundColor: COLORS.primary,
    borderColor: COLORS.primary,
  },
  radiusPillText: { color: COLORS.textMuted, fontSize: 12.5, fontWeight: '700' },
  radiusPillTextActive: { color: COLORS.onPrimary },

  radarArea: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  radar: {
    width: RADAR_SIZE,
    height: RADAR_SIZE,
    alignItems: 'center',
    justifyContent: 'center',
  },
  ring: {
    position: 'absolute',
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
  },
  ping: {
    position: 'absolute',
    width: RADAR_SIZE,
    height: RADAR_SIZE,
    borderRadius: RADAR_CENTER,
    borderWidth: 1.5,
    borderColor: COLORS.primary,
  },
  youDot: {
    position: 'absolute',
    width: 26,
    height: 26,
    borderRadius: 13,
    backgroundColor: COLORS.primary,
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 5,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.7,
    shadowRadius: 8,
  },
  blipWrap: {
    position: 'absolute',
    width: 32,
    height: 32,
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 4,
  },
  blipHalo: {
    position: 'absolute',
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: COLORS.primary,
    opacity: 0.25,
  },
  blip: {
    width: 26,
    height: 26,
    borderRadius: 13,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.4,
    shadowRadius: 4,
  },

  hint: {
    color: COLORS.textMuted,
    fontSize: 12,
    fontWeight: '600',
    marginTop: 18,
  },

  detailCard: {
    position: 'absolute',
    bottom: 12,
    width: RADAR_SIZE,
    backgroundColor: COLORS.surface,
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
    borderRadius: 18,
    padding: 16,
  },
  detailClose: {
    position: 'absolute',
    top: 10,
    right: 10,
    zIndex: 2,
  },
  detailHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 14,
  },
  detailIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  detailTitle: { color: COLORS.text, fontSize: 15, fontWeight: '700' },
  detailMeta: { color: COLORS.textMuted, fontSize: 12, fontWeight: '600', marginTop: 2 },
  detailScore: { fontSize: 22, fontWeight: '800' },

  mapBtn: {
    flexDirection: 'row',
    paddingVertical: 12,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  mapBtnText: { color: COLORS.onPrimary, fontSize: 14, fontWeight: '700' },

  emptyState: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 30 },
  emptyText: { color: COLORS.textMuted, fontSize: 14, textAlign: 'center', marginTop: 14, lineHeight: 20 },
});
