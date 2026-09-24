/**
 * Unit tests for mobile_app/src/theme/colors.js — getScoreColor() and
 * getScoreMood(). Both are pure functions (no React Native / Expo
 * imports), so they're required directly with no mocking needed.
 */
const { getScoreColor, getScoreMood, WELLBEING_GRADIENT } = require('../../../mobile_app/src/theme/colors');

describe('getScoreColor', () => {
  test('returns the exact gradient color at each defined stop', () => {
    WELLBEING_GRADIENT.forEach(({ stop, color }) => {
      expect(getScoreColor(stop).toLowerCase()).toBe(color.toLowerCase());
    });
  });

  test('interpolates a color at the midpoint between two stops', () => {
    // Between stop 0.75 (#7FA06A) and stop 1.0 (#6E8B57): midpoint 0.875
    // should be roughly halfway between those two hex values.
    const mid = getScoreColor(0.875);
    expect(mid.toLowerCase()).not.toBe('#7fa06a');
    expect(mid.toLowerCase()).not.toBe('#6e8b57');
    expect(mid).toMatch(/^#[0-9a-f]{6}$/i);
  });

  test('clamps scores above 1.0 to the top-of-gradient color', () => {
    expect(getScoreColor(1.5).toLowerCase()).toBe('#6e8b57');
  });

  test('clamps scores below 0.0 to the bottom-of-gradient color', () => {
    expect(getScoreColor(-0.5).toLowerCase()).toBe('#c1443a');
  });

  test('defaults to the midpoint (0.5) color when score is null/undefined', () => {
    const defaultColor = getScoreColor(undefined);
    expect(defaultColor).toBe(getScoreColor(0.5));
  });
});

describe('getScoreMood', () => {
  test.each([
    [0.0, 'Distressing'],
    [0.34, 'Distressing'],
    [0.35, 'Mixed'],
    [0.54, 'Mixed'],
    [0.55, 'Calm'],
    [0.74, 'Calm'],
    [0.75, 'Restorative'],
    [1.0, 'Restorative'],
  ])('score %f -> %s', (score, expected) => {
    expect(getScoreMood(score)).toBe(expected);
  });

  test('defaults to the 0.5 band when score is null/undefined', () => {
    expect(getScoreMood(undefined)).toBe(getScoreMood(0.5));
  });
});
