/**
 * Unit tests for mobile_app/src/utils/soundHuntStore.js.
 *
 * These functions are thin fetch() wrappers around the Flask API, so
 * "unit" here means: mock global.fetch, and assert (a) the correct
 * request is sent, and (b) the response/error shape is handled
 * correctly — never a real network call.
 */
const { getCompletedChallenges, completeChallenge } = require('../../../mobile_app/src/utils/soundHuntStore');

beforeEach(() => {
  global.fetch = jest.fn();
  // getCompletedChallenges() intentionally logs failures via console.error
  // (see soundHuntStore.js) — expected app behaviour, not a test failure.
  // Silenced here so a passing test doesn't print a scary-looking error.
  jest.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => {
  jest.resetAllMocks();
});

describe('getCompletedChallenges', () => {
  test('requests the completions endpoint and returns the completed list', async () => {
    global.fetch.mockResolvedValue({
      json: () => Promise.resolve({ completed: ['rain', 'silence'] }),
    });

    const result = await getCompletedChallenges();

    expect(global.fetch).toHaveBeenCalledWith('http://127.0.0.1:5001/api/sound_hunt/completions');
    expect(result).toEqual(['rain', 'silence']);
  });

  test('returns an empty array when the response has no completed field', async () => {
    global.fetch.mockResolvedValue({ json: () => Promise.resolve({}) });
    const result = await getCompletedChallenges();
    expect(result).toEqual([]);
  });

  test('swallows a network failure and returns an empty array rather than throwing', async () => {
    global.fetch.mockRejectedValue(new Error('network down'));
    const result = await getCompletedChallenges();
    expect(result).toEqual([]);
  });

  test('swallows a backend error response and returns an empty array', async () => {
    global.fetch.mockResolvedValue({ json: () => Promise.resolve({ error: 'db locked' }) });
    const result = await getCompletedChallenges();
    expect(result).toEqual([]);
  });
});

describe('completeChallenge', () => {
  test('POSTs the challenge_id as JSON with the correct headers', async () => {
    global.fetch.mockResolvedValue({ json: () => Promise.resolve({ completed: ['rain'] }) });

    await completeChallenge('rain');

    expect(global.fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:5001/api/sound_hunt/complete',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: JSON.stringify({ challenge_id: 'rain' }),
      },
    );
  });

  test('resolves with the completed value from the response', async () => {
    global.fetch.mockResolvedValue({ json: () => Promise.resolve({ completed: ['rain'] }) });
    const result = await completeChallenge('rain');
    expect(result).toEqual(['rain']);
  });

  test('throws when the backend responds with an error, instead of swallowing it', async () => {
    global.fetch.mockResolvedValue({ json: () => Promise.resolve({ error: 'challenge_id is required' }) });
    await expect(completeChallenge(undefined)).rejects.toThrow('challenge_id is required');
  });
});
