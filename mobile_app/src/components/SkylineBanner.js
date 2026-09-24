import React from 'react';
import { View, StyleSheet, Dimensions } from 'react-native';
import Svg, { Defs, LinearGradient, Stop, Rect, Circle as SvgCircle, Path } from 'react-native-svg';
import { COLORS } from '../theme/colors';

const { width } = Dimensions.get('window');

// The shared header motif: a dusk city skyline with leaf sprigs over it.
// `height` scales it down for screens needing a compact strip.
export default function SkylineBanner({ height = 130, leaves = true, leafSize }) {
  const scale = height / 130;
  const resolvedLeafSize = leafSize ?? Math.max(60, 94 * scale);

  const buildings = [
    { x: 0, y: 46, w: 26, h: 70, fade: 'a' },
    { x: 28, y: 30, w: 20, h: 86, fade: 'b' },
    { x: 50, y: 52, w: 18, h: 64, fade: 'a' },
    { x: 70, y: 18, w: 24, h: 98, fade: 'b' },
    { x: 96, y: 40, w: 16, h: 76, fade: 'a' },
    { x: width - 100, y: 44, w: 18, h: 72, fade: 'b' },
    { x: width - 80, y: 60, w: 16, h: 56, fade: 'a' },
    { x: width - 62, y: 24, w: 22, h: 92, fade: 'b' },
    { x: width - 38, y: 50, w: 18, h: 66, fade: 'a' },
    { x: width - 18, y: 36, w: 18, h: 80, fade: 'b' },
  ];
  const windows = [
    [4, 52], [14, 60], [4, 70],
    [74, 26], [84, 36], [74, 48], [84, 58],
    [width - 58, 32], [width - 48, 42], [width - 58, 54],
    [width - 14, 44], [width - 14, 60],
  ];

  return (
    <View style={[styles.bannerWrap, { height }]}>
      <Svg width={width} height={height} viewBox={`0 0 ${width} 130`} preserveAspectRatio="xMidYMax slice">
        <Defs>
          <LinearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
            <Stop offset="0%" stopColor="#E9C98A" />
            <Stop offset="55%" stopColor="#E9C98A" />
            <Stop offset="100%" stopColor={COLORS.background} />
          </LinearGradient>
          <LinearGradient id="bfadeA" x1="0" y1="0" x2="0" y2="1">
            <Stop offset="0%" stopColor="#79695A" stopOpacity="1" />
            <Stop offset="100%" stopColor="#79695A" stopOpacity="0" />
          </LinearGradient>
          <LinearGradient id="bfadeB" x1="0" y1="0" x2="0" y2="1">
            <Stop offset="0%" stopColor="#6C5D50" stopOpacity="1" />
            <Stop offset="100%" stopColor="#6C5D50" stopOpacity="0" />
          </LinearGradient>
        </Defs>
        <Rect width={width} height={130} fill="url(#sky)" />
        <SvgCircle cx={width - 72} cy={30} r={28} fill="#F3DBA2" opacity={0.9} />
        {buildings.map((b, i) => (
          <Rect key={i} x={b.x} y={b.y} width={b.w} height={b.h} fill={`url(#bfade${b.fade === 'a' ? 'A' : 'B'})`} />
        ))}
        {windows.map(([wx, wy], i) => (
          <Rect key={i} x={wx} y={wy} width={4} height={4} fill="#F3DBA2" opacity={0.85} />
        ))}
      </Svg>

      {leaves && (
        <>
          <Svg width={resolvedLeafSize} height={resolvedLeafSize} viewBox="0 0 94 94" style={[styles.leafLeft, { top: height * 0.6 }]}>
            <Path d="M8 8 C42 8 56 30 56 64 C28 64 8 42 8 8Z" fill="#7EA163" opacity={0.9} />
            <Path d="M8 8 C36 24 46 44 48 62" stroke="#5C7A46" strokeWidth={1.8} fill="none" />
          </Svg>
          <Svg width={resolvedLeafSize} height={resolvedLeafSize} viewBox="0 0 94 94" style={[styles.leafRight, { transform: [{ scaleX: -1 }] }]}>
            <Path d="M8 8 C42 8 56 30 56 64 C28 64 8 42 8 8Z" fill="#9AB37E" opacity={0.85} />
            <Path d="M8 8 C36 24 46 44 48 62" stroke="#5C7A46" strokeWidth={1.8} fill="none" />
          </Svg>
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  bannerWrap: {
    width,
    position: 'relative',
  },
  leafLeft: {
    position: 'absolute',
    left: -10,
  },
  leafRight: {
    position: 'absolute',
    bottom: -12,
    right: -12,
  },
});
