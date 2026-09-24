// Completions live on the Flask backend, so progress survives reinstalls.
const API_BASE = "http://127.0.0.1:5001";

export async function getCompletedChallenges() {
  try {
    const res = await fetch(`${API_BASE}/api/sound_hunt/completions`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    return data.completed || [];
  } catch (err) {
    console.error('Failed to load Sound Hunt completions:', err);
    return [];
  }
}

// The server re-classifies the audio itself (models/sound_hunt.py), so the
// recording is sent along with the challenge_id, not just the id.
export async function completeChallenge(challengeId, base64Audio) {
  const res = await fetch(`${API_BASE}/api/sound_hunt/complete`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
    body: JSON.stringify({ challenge_id: challengeId, audio: base64Audio }),
  });
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data.completed;
}
