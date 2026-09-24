import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import { View, StyleSheet, Dimensions, Text, TouchableOpacity, ScrollView, TextInput, SafeAreaView, ActivityIndicator } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import MapView, { Marker, Callout } from 'react-native-maps';
import * as Location from 'expo-location';
import { Feather } from '@expo/vector-icons';
import { COLORS, getScoreColor } from '../theme/colors';
import { getMapPins } from '../utils/mapStore';
import { EMOTION_DISPLAY } from '../utils/diaryStore';
import SkylineBanner from '../components/SkylineBanner';

const { width, height } = Dimensions.get('window');

// Same icon+color pairing Diary uses, so an emotion reads the same everywhere.
const EMOTIONS = [
  { name: 'Calm', icon: 'sun', color: EMOTION_DISPLAY.Calm.color },
  { name: 'Content', icon: 'smile', color: EMOTION_DISPLAY.Content.color },
  { name: 'Anxious', icon: 'cloud-drizzle', color: EMOTION_DISPLAY.Anxious.color },
  { name: 'Stressed', icon: 'cloud-lightning', color: EMOTION_DISPLAY.Stressed.color },
  { name: 'Annoyed', icon: 'frown', color: EMOTION_DISPLAY.Annoyed.color },
];

// Best -> worst (Restorative -> Distressing).
const EMOTION_LEGEND_ORDER = ['Content', 'Calm', 'Anxious', 'Annoyed', 'Stressed'];

// Fallback start point (F-8, Islamabad) if location is denied or unavailable.
const FALLBACK_REGION = {
  latitude: 33.7098,
  longitude: 73.0555,
  latitudeDelta: 0.0922,
  longitudeDelta: 0.0421,
};

// How far a pin can be and still count as "nearby" for auto-fitting the
// camera, so one stray distant pin can't zoom the map out to the whole Earth.
const NEARBY_RADIUS_KM = 50;

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

// Minimum zoom span: roughly an 8-9km neighbourhood view.
const DEFAULT_DELTA = 0.08;

// fitToCoordinates zooms to the exact bounding box, which collapses to
// street level for a single nearby pin. Computing the region here lets the
// MIN_DELTA floor apply.
function computeRegion(coordsList) {
  const lats = coordsList.map(c => c.latitude);
  const lngs = coordsList.map(c => c.longitude);
  const minLat = Math.min(...lats), maxLat = Math.max(...lats);
  const minLng = Math.min(...lngs), maxLng = Math.max(...lngs);
  return {
    latitude: (minLat + maxLat) / 2,
    longitude: (minLng + maxLng) / 2,
    latitudeDelta: Math.max((maxLat - minLat) * 1.6, DEFAULT_DELTA),
    longitudeDelta: Math.max((maxLng - minLng) * 1.6, DEFAULT_DELTA),
  };
}

export default function MapScreen({ route, navigation }) {
  const mapRef = useRef(null);
  const markerRefs = useRef({}); // pin id -> Marker ref, so a specific pin's callout can be opened programmatically
  const [pins, setPins] = useState([]);
  const [mapReady, setMapReady] = useState(false);
  const [initialRegion, setInitialRegion] = useState(null);
  const [userLocation, setUserLocation] = useState(null);
  const [focusedPinId, setFocusedPinId] = useState(null); // pin whose callout was opened programmatically (Find on Map)
  const [activeFilters, setActiveFilters] = useState(EMOTIONS.map(e => e.name));
  const [searchQuery, setSearchQuery] = useState('');
  const [searchOpen, setSearchOpen] = useState(false);

  // Centre on the user's real location, falling back to F-8 if unavailable.
  useEffect(() => {
    (async () => {
      try {
        const perm = await Location.requestForegroundPermissionsAsync();
        if (perm.status !== 'granted') {
          setInitialRegion(FALLBACK_REGION);
          return;
        }
        // The Simulator throws a transient kCLErrorLocationUnknown; one
        // retry resolves most cases without falling back.
        let position;
        try {
          position = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Highest });
        } catch (firstErr) {
          console.warn('Location fix failed once, retrying:', firstErr.message);
          await new Promise(resolve => setTimeout(resolve, 1200));
          position = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Highest });
        }
        const coords = { latitude: position.coords.latitude, longitude: position.coords.longitude };
        setUserLocation(coords);
        setInitialRegion({ ...coords, latitudeDelta: DEFAULT_DELTA, longitudeDelta: DEFAULT_DELTA });
      } catch (err) {
        // warn, not error: LogBox turns errors into a full-screen overlay.
        console.warn('Failed to get current location for map:', err.message);
        setInitialRegion(FALLBACK_REGION);
      }
    })();
  }, []);

  // Reload on focus so a pin saved from RecordScreen appears immediately.
  useFocusEffect(
    useCallback(() => {
      let isActive = true;
      (async () => {
        const loaded = await getMapPins();
        if (isActive) setPins(loaded);
      })();
      return () => { isActive = false; };
    }, [])
  );

  // Filtered by the active emotion pills and, if the search box holds a
  // number, by score. Both "0.7" and "70" match a pin scored 0.70.
  const visiblePins = useMemo(() => {
    let result = pins.filter(p => activeFilters.includes(p.emotion));

    const query = searchQuery.trim();
    if (query !== '') {
      const queryNum = parseFloat(query);
      if (!isNaN(queryNum)) {
        result = result.filter(p => {
          if (typeof p.wellbeingScore !== 'number') return false;
          const asDecimal = Math.abs(p.wellbeingScore - queryNum) < 0.005;
          const asPercent = Math.abs(p.wellbeingScore * 100 - queryNum) < 0.5;
          return asDecimal || asPercent;
        });
      }
    }
    return result;
  }, [pins, activeFilters, searchQuery]);

  // Fit the camera to pins within NEARBY_RADIUS_KM only. Farther pins still
  // render, they just don't drag the camera out to include them.
  useEffect(() => {
    if (!mapReady || !mapRef.current) return;
    // A specific spot was requested, so skip the usual nearby-pins fit.
    if (route?.params?.focusLocation) return;

    const nearbyPins = userLocation
      ? visiblePins.filter(p => haversineDistanceKm(userLocation, p.loc) <= NEARBY_RADIUS_KM)
      : visiblePins;

    if (nearbyPins.length > 0) {
      const coords = userLocation ? [userLocation, ...nearbyPins.map(p => p.loc)] : nearbyPins.map(p => p.loc);
      mapRef.current.animateToRegion(computeRegion(coords), 500);
    }
  }, [mapReady, visiblePins, userLocation, route?.params?.focusLocation]);

  // Jump to a spot passed in via navigation (Quiet Spots' "Find on Map").
  // The param is cleared after use so returning to the tab doesn't re-jump;
  // if a pin id came with it, its callout opens once the camera settles.
  useEffect(() => {
    const target = route?.params?.focusLocation;
    const targetPinId = route?.params?.focusPinId;
    if (mapReady && mapRef.current && target) {
      mapRef.current.animateToRegion({ ...target, latitudeDelta: DEFAULT_DELTA, longitudeDelta: DEFAULT_DELTA }, 500);
      if (targetPinId) {
        setTimeout(() => {
          markerRefs.current[targetPinId]?.showCallout();
          setFocusedPinId(targetPinId);
        }, 600);
      }
      navigation?.setParams({ focusLocation: undefined, focusPinId: undefined });
    }
  }, [mapReady, route?.params?.focusLocation, route?.params?.focusPinId]);

  // A callout opened programmatically (via showCallout() above, from
  // Quiet Spots' "Find on Map") isn't wired into the native tap-elsewhere-
  // to-dismiss gesture the way a normally-tapped callout is — so it can
  // get stuck open. Tracking which pin's callout we opened and explicitly
  // hiding it on any other map tap restores the behavior a user expects
  // from tapping a marker directly.
  const dismissFocusedCallout = () => {
    if (focusedPinId) {
      markerRefs.current[focusedPinId]?.hideCallout();
      setFocusedPinId(null);
    }
  };

  const recenterOnUser = () => {
    if (userLocation && mapRef.current) {
      mapRef.current.animateToRegion({ ...userLocation, latitudeDelta: DEFAULT_DELTA, longitudeDelta: DEFAULT_DELTA }, 400);
    }
  };

  const toggleFilter = (emotion) => {
    if (activeFilters.includes(emotion)) {
      setActiveFilters(prev => prev.filter(e => e !== emotion));
    } else {
      setActiveFilters(prev => [...prev, emotion]);
    }
  };

  const handleZoom = (direction) => {
    if (mapRef.current) {
      mapRef.current.getCamera().then((cam) => {
        cam.altitude = direction === 'in' ? cam.altitude / 2 : cam.altitude * 2;
        cam.zoom = direction === 'in' ? (cam.zoom || 14) + 1 : (cam.zoom || 14) - 1;
        mapRef.current.animateCamera(cam);
      });
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.headerBlock}>
        <View style={styles.bannerBleed}>
          <SkylineBanner />
        </View>
        <View style={styles.headerTop}>
          <Text style={styles.headerTitle}>Map</Text>
          <TouchableOpacity style={styles.searchToggle} onPress={() => setSearchOpen((v) => !v)}>
            <Feather name="search" size={16} color={searchOpen ? COLORS.primary : COLORS.textMuted} />
          </TouchableOpacity>
        </View>

        {searchOpen && (
          <TextInput
            style={styles.searchInput}
            placeholder="Score, e.g. 70 or 0.7"
            placeholderTextColor={COLORS.textMuted}
            value={searchQuery}
            onChangeText={setSearchQuery}
            keyboardType="numeric"
            autoFocus
          />
        )}

        <Text style={styles.filterHeading}>FILTER BY EMOTION</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.filterScroll}>
          {EMOTIONS.map(emo => {
            const isActive = activeFilters.includes(emo.name);
            return (
              <TouchableOpacity
                key={emo.name}
                style={[
                  styles.filterPill,
                  { borderColor: emo.color },
                  isActive && { backgroundColor: emo.color },
                ]}
                onPress={() => toggleFilter(emo.name)}
              >
                <Feather name={emo.icon} size={12} color={isActive ? COLORS.onPrimary : emo.color} style={{ marginRight: 5 }} />
                <Text style={[styles.filterPillText, { color: isActive ? COLORS.onPrimary : emo.color }]}>{emo.name}</Text>
              </TouchableOpacity>
            );
          })}
        </ScrollView>

        {/* Same worst -> best ordering as the emotion-ranking work in
            Analyse/Step 4 (Stressed -> Annoyed -> Anxious -> Calm ->
            Content), built from each emotion's own color rather than the
            abstract wellbeing gradient, so the legend reads directly off
            the filter pills above it. */}
        <View style={styles.legendRow}>
          <View style={styles.legendGradient}>
            {EMOTION_LEGEND_ORDER.map((name, i) => (
              <View key={i} style={[styles.legendSegment, { backgroundColor: EMOTION_DISPLAY[name].color }]} />
            ))}
          </View>
          <Text style={styles.legendLabelLeft}>Restorative</Text>
          <Text style={styles.legendLabelRight}>Distressing</Text>
        </View>

        {pins.length === 0 ? (
          <View style={styles.bannerBlock}>
            <Feather name="map-pin" size={13} color={COLORS.warning} />
            <Text style={styles.bannerText}>No reports yet — be the first</Text>
          </View>
        ) : visiblePins.length === 0 ? (
          <View style={styles.bannerBlock}>
            <Feather name="filter" size={13} color={COLORS.warning} />
            <Text style={styles.bannerText}>No matches for this filter</Text>
          </View>
        ) : null}
      </View>

      {/* Map sits naturally below the header without overlapping */}
      <View style={styles.mapContainer}>
        {!initialRegion ? (
          <View style={[styles.map, styles.mapLoading]}>
            <ActivityIndicator size="large" color={COLORS.primary} />
            <Text style={styles.mapLoadingText}>Finding your location…</Text>
          </View>
        ) : (
        <MapView
          ref={mapRef}
          style={styles.map}
          initialRegion={initialRegion}
          userInterfaceStyle="dark"
          showsUserLocation={true}
          onMapReady={() => setMapReady(true)}
          onPress={dismissFocusedCallout}
          onUserLocationChange={(event) => {
            // The map's own native location fix — updates userLocation
            // whenever the blue dot itself gets/updates a real position,
            // independent of the one-shot getCurrentPositionAsync() call
            // above. That call can fail (a known Simulator GPS flake)
            // while the map still eventually gets a fix on its own; the
            // recenter button and nearby-pin fit should track whichever
            // source actually has a location, not just the first one.
            const coords = event.nativeEvent.coordinate;
            if (coords) setUserLocation({ latitude: coords.latitude, longitude: coords.longitude });
          }}
        >
          {visiblePins.map(pin => {
            // The pin's identity color is its EMOTION — the same canonical
            // color used everywhere else in the app (filter pills, legend,
            // Diary, Analyse). getScoreColor is reserved for the score
            // VALUE itself (the percentage text below), never for anything
            // that represents "which emotion" — mixing the two here (an
            // earlier version colored the marker/dot by score) is exactly
            // what caused an Annoyed pin to render orange instead of pink.
            const emotionColor = EMOTION_DISPLAY[pin.emotion]?.color || COLORS.textMuted;
            const scoreColor = typeof pin.wellbeingScore === 'number' ? getScoreColor(pin.wellbeingScore) : COLORS.textMuted;
            return (
              <Marker
                key={pin.id}
                ref={(ref) => { markerRefs.current[pin.id] = ref; }}
                coordinate={pin.loc}
                pinColor={emotionColor}
                onPress={() => setFocusedPinId(pin.id)}
              >
                <Callout tooltip>
                  <View style={styles.calloutContainer}>
                    <View style={styles.calloutHeaderRow}>
                      <View style={[styles.calloutDot, { backgroundColor: emotionColor }]} />
                      <Text style={[styles.calloutTitle, { color: emotionColor }]}>{pin.emotion}</Text>
                    </View>
                    <Text style={styles.calloutDesc}>{pin.desc}</Text>
                    {pin.sceneLabel && (
                      <Text style={styles.calloutScene}>{pin.sceneLabel}</Text>
                    )}
                    {typeof pin.wellbeingScore === 'number' && (
                      <Text style={[styles.calloutScore, { color: scoreColor }]}>{Math.round(pin.wellbeingScore * 100)}%</Text>
                    )}
                  </View>
                </Callout>
              </Marker>
            );
          })}
        </MapView>
        )}

        {/* Map Controls */}
        <View style={styles.mapControls}>
          <TouchableOpacity style={styles.controlBtn} onPress={() => handleZoom('in')}>
            <Feather name="plus" size={18} color={COLORS.text} />
          </TouchableOpacity>
          <View style={styles.controlDivider} />
          <TouchableOpacity style={styles.controlBtn} onPress={() => handleZoom('out')}>
            <Feather name="minus" size={18} color={COLORS.text} />
          </TouchableOpacity>
        </View>

        {userLocation && (
          <TouchableOpacity style={styles.recenterBtn} onPress={recenterOnUser}>
            <Feather name="navigation" size={18} color={COLORS.primary} />
          </TouchableOpacity>
        )}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.background },
  mapContainer: { flex: 1 },
  mapLoading: { alignItems: 'center', justifyContent: 'center', backgroundColor: COLORS.background },
  mapLoadingText: { color: COLORS.textMuted, fontSize: 13, marginTop: 12 },
  map: { flex: 1, width: width },
  headerBlock: {
    backgroundColor: COLORS.background,
    paddingHorizontal: 20,
    paddingTop: 0,
    paddingBottom: 14,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.cardBorder,
  },
  bannerBleed: {
    marginHorizontal: -20,
    marginBottom: 10,
  },
  headerTop: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  headerTitle: { color: COLORS.text, fontSize: 20, fontWeight: '800' },
  searchToggle: {
    width: 34,
    height: 34,
    borderRadius: 17,
    backgroundColor: COLORS.surface,
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
    alignItems: 'center',
    justifyContent: 'center',
  },

  searchInput: {
    backgroundColor: COLORS.surface,
    color: COLORS.text,
    paddingHorizontal: 14,
    paddingVertical: 9,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
    marginBottom: 12,
    fontSize: 13.5,
  },

  filterHeading: {
    color: COLORS.secondary,
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 1.3,
    marginBottom: 10,
  },
  filterScroll: { flexDirection: 'row', marginBottom: 12 },
  filterPill: {
    backgroundColor: COLORS.surface,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 11,
    paddingVertical: 6,
    borderRadius: 16,
    marginRight: 8,
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
  },
  filterPillText: { fontSize: 12, fontWeight: '600' },

  legendRow: {
    marginBottom: 10,
  },
  legendGradient: {
    flexDirection: 'row',
    height: 5,
    borderRadius: 3,
    overflow: 'hidden',
    marginBottom: 4,
  },
  legendSegment: {
    flex: 1,
  },
  legendLabelLeft: { position: 'absolute', left: 0, top: 9, color: COLORS.textMuted, fontSize: 10 },
  legendLabelRight: { position: 'absolute', right: 0, top: 9, color: COLORS.textMuted, fontSize: 10 },

  bannerBlock: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(234, 179, 8, 0.1)',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: 'rgba(234, 179, 8, 0.25)',
    marginTop: 14,
  },
  bannerText: { color: COLORS.warning, fontSize: 12, fontWeight: '600', marginLeft: 7 },

  mapControls: {
    position: 'absolute',
    left: 20,
    bottom: 40,
    backgroundColor: COLORS.surfaceElevated,
    borderRadius: 10,
    width: 40,
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
  },
  controlBtn: {
    height: 40,
    alignItems: 'center',
    justifyContent: 'center',
  },
  recenterBtn: {
    position: 'absolute',
    right: 20,
    bottom: 40,
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: COLORS.surfaceElevated,
    borderWidth: 1,
    borderColor: COLORS.cardBorder,
    alignItems: 'center',
    justifyContent: 'center',
  },
  controlDivider: {
    height: 1,
    backgroundColor: COLORS.cardBorder,
    width: '100%',
  },

  calloutContainer: {
    backgroundColor: COLORS.background,
    padding: 10,
    borderRadius: 10,
    borderColor: COLORS.cardBorder,
    borderWidth: 1,
    width: 160,
  },
  calloutHeaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  calloutDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 6,
  },
  calloutTitle: { fontWeight: 'bold', fontSize: 15, color: COLORS.text },
  calloutDesc: { color: COLORS.textMuted, fontSize: 12, marginTop: 5 },
  calloutScene: { color: COLORS.textMuted, fontSize: 11, marginTop: 2, fontStyle: 'italic' },
  calloutScore: { fontSize: 13, marginTop: 4, fontWeight: '700' },
});
