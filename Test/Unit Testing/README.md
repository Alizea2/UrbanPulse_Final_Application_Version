# UrbanPulse — Unit Tests

## Backend (`backend/`)

Python, using **pytest**. Tests the deterministic logic behind the AI pipeline
(never the models themselves — those are covered separately under
`Tests/model_evaluation/`).

- `test_wellbeing_scorer.py` — `combine_scores()` scoring formula
- `test_db.py` — `_haversine_m()` distance math and the map-pin
  replace-within-radius rule (runs against a temp SQLite file, never the
  real `urbanpulse.db`)
- `test_api.py` — Flask routes via `test_client()`, with the 4 heavy AI
  models stubbed out so tests run in milliseconds without needing
  torch/tensorflow/transformers or real model weights

Run from the project root:

```bash
python3 -m pytest "Unit Testing/backend" -v
```

## Frontend (`frontend/`)

JavaScript, using **Jest**. Has its own `package.json`/`node_modules`
(installed here rather than inside `mobile_app`, since these are plain
Node/Jest tests with no React Native runtime dependency).

- `colors.test.js` — `getScoreColor()` / `getScoreMood()` gradient logic
- `soundHuntStore.test.js` — `getCompletedChallenges()` / `completeChallenge()`,
  with `global.fetch` mocked (no real network calls)

Run:

```bash
cd "Unit Testing/frontend"
npx jest
```
