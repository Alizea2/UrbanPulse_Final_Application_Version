// Diary entries live on the Flask backend (see models/db.py), not in local
// storage, so they persist across reinstalls and devices.
const API_BASE = "http://127.0.0.1:5001";

// Re-export for existing imports; defined in theme/colors.js.
export { EMOTION_COLORS as EMOTION_DISPLAY } from '../theme/colors';

export async function getDiaryEntries() {
  try {
    const res = await fetch(`${API_BASE}/api/diary/entries`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    return data.entries || [];
  } catch (err) {
    console.error('Failed to load diary entries:', err);
    return [];
  }
}

export async function saveDiaryEntry(entry) {
  const res = await fetch(`${API_BASE}/api/diary/entries`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
    body: JSON.stringify(entry),
  });
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data.entry;
}
