// Map pins live on the Flask backend, making the map a shared dataset.
// The 80m replace-on-proximity rule is enforced server-side (models/db.py).
const API_BASE = "http://127.0.0.1:5001";

export async function getMapPins() {
  try {
    const res = await fetch(`${API_BASE}/api/map/pins`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    return data.pins || [];
  } catch (err) {
    console.error('Failed to load map pins:', err);
    return [];
  }
}

export async function saveMapPin(pin) {
  const res = await fetch(`${API_BASE}/api/map/pins`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
    body: JSON.stringify(pin),
  });
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data.pin;
}
