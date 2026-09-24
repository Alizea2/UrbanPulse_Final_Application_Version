import React, { useEffect } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Svg, { Circle } from 'react-native-svg';
import Animated, { useSharedValue, useAnimatedProps, withTiming, Easing } from 'react-native-reanimated';
import { COLORS, getScoreColor, getScoreMood } from '../theme/colors';

const AnimatedCircle = Animated.createAnimatedComponent(Circle);

/**
 * Animated Wellbeing Score ring, coloured off WELLBEING_GRADIENT, with the
 * score and mood word at the centre. Used on Home and Analyse.
 */
export default function ScoreRing({ score, hasData = true, size = 176, strokeWidth = 14 }) {
  // The stroke is drawn centred on the path, so half of it sits outside
  // the radius -- subtracting it keeps the ring inside `size`.
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const progress = useSharedValue(0);

  // Animates to the new score whenever it changes.
  useEffect(() => {
    progress.value = withTiming(hasData ? (score ?? 0) : 0, { duration: 900, easing: Easing.out(Easing.cubic) });
  }, [score, hasData]);

  // The arc is one dash the length of the full circle; shortening the
  // offset is what reveals it.
  const animatedProps = useAnimatedProps(() => ({
    strokeDashoffset: circumference * (1 - progress.value),
  }));

  // Grey until there is a real score to show.
  const color = hasData ? getScoreColor(score) : COLORS.textMuted;

  return (
    <View style={[styles.wrap, { width: size, height: size }]}>
      <Svg width={size} height={size}>
        {/* The full track the animated arc is drawn over. */}
        <Circle cx={size / 2} cy={size / 2} r={radius} stroke={COLORS.surfaceElevated} strokeWidth={strokeWidth} fill="none" />
        <AnimatedCircle
          cx={size / 2} cy={size / 2} r={radius}
          stroke={color} strokeWidth={strokeWidth} fill="none"
          strokeDasharray={circumference}
          strokeLinecap="round"
          rotation="-90"   /* start at 12 o'clock, not 3 */
          originX={size / 2}
          originY={size / 2}
          animatedProps={animatedProps}
        />
      </Svg>
      <View style={styles.center}>
        <Text style={[styles.score, { color, fontSize: size * 0.19 }]}>{hasData ? score.toFixed(2) : '—'}</Text>
        <Text style={styles.mood}>{hasData ? getScoreMood(score) : 'No data yet'}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  center: {
    position: 'absolute',
    justifyContent: 'center',
    alignItems: 'center',
  },
  score: {
    fontWeight: '800',
  },
  mood: {
    color: COLORS.textMuted,
    fontSize: 12,
    marginTop: 2,
  },
});
