// "Botanical Dusk": a light sage/cream base with dusk-amber as the accent.
// Every screen reads colours through these tokens, so a theme change
// happens here and nowhere else.
export const COLORS = {
  background: '#EDEFE3',
  // Two depth tiers above `background`, for layering cards within cards.
  surface: '#F3F5EB',
  surfaceElevated: '#DDE3CB',
  card: '#F3F5EB',
  cardBorder: '#D6D9C4',
  primary: '#6E8B57', // sage green — the calm you're finding
  secondary: '#B8863F', // dusk amber — the city, muted enough to read on a light card
  text: '#3A3F2E',
  textMuted: '#7C8265',
  danger: '#B84A3A',
  warning: '#C9922E',
  // For anything on a solid `primary` fill: sage is mid-tone, so it needs
  // light text rather than the usual dark `text`.
  onPrimary: '#F2F5EA',
};

// Single source of truth for the 5 emotion states -> icon + colour.
export const EMOTION_COLORS = {
  Calm: { icon: 'sun', color: '#7FA06A' },
  Content: { icon: 'smile', color: '#5C87A0' },
  Anxious: { icon: 'cloud-drizzle', color: '#C9922E' },
  Stressed: { icon: 'cloud-lightning', color: '#B84A3A' },
  Annoyed: { icon: 'frown', color: '#A8577A' },
};

// Every Wellbeing Score on every screen reads off this same
// distressing -> restorative scale.
export const WELLBEING_GRADIENT = [
  { stop: 0.0, color: '#C1443A' }, // distressing
  { stop: 0.35, color: '#D98E3F' },
  { stop: 0.55, color: '#C9A227' },
  { stop: 0.75, color: '#7FA06A' },
  { stop: 1.0, color: '#6E8B57' }, // restorative
];

const hexToRgb = (hex) => {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};

const rgbToHex = (r, g, b) =>
  '#' + [r, g, b].map((v) => Math.round(v).toString(16).padStart(2, '0')).join('');

/** Interpolates a Wellbeing Score (0-1) along WELLBEING_GRADIENT. Used
 * anywhere a score needs to become a color instead of a raw number. */
export function getScoreColor(score) {
  const clamped = Math.max(0, Math.min(1, score ?? 0.5));
  for (let i = 0; i < WELLBEING_GRADIENT.length - 1; i++) {
    const a = WELLBEING_GRADIENT[i];
    const b = WELLBEING_GRADIENT[i + 1];
    if (clamped >= a.stop && clamped <= b.stop) {
      const t = (clamped - a.stop) / (b.stop - a.stop);
      const [r1, g1, b1] = hexToRgb(a.color);
      const [r2, g2, b2] = hexToRgb(b.color);
      return rgbToHex(r1 + (r2 - r1) * t, g1 + (g2 - g1) * t, b1 + (b2 - b1) * t);
    }
  }
  return WELLBEING_GRADIENT[WELLBEING_GRADIENT.length - 1].color;
}

/** One-word mood label for a score, matching the same gradient bands —
 * shown alongside the color so the number is never the only signal. */
export function getScoreMood(score) {
  const s = score ?? 0.5;
  if (s < 0.35) return 'Distressing';
  if (s < 0.55) return 'Mixed';
  if (s < 0.75) return 'Calm';
  return 'Restorative';
}
