"""Tests how the Flask backend handles simultaneous requests, using the fast
/api/analyze endpoint so the result reflects concurrency, not inference time.
"""
import time
import sys
import concurrent.futures
import requests

API_BASE = "http://127.0.0.1:5001"
AUDIO_PATH = "test.wav"
CONCURRENCY_LEVELS = [1, 2, 4, 8]   # workers sending at once
REQUESTS_PER_LEVEL = 8              # total requests per level


# Times a single request end to end, from the client's side.
def one_request():
    with open(AUDIO_PATH, "rb") as f:
        start = time.perf_counter()
        r = requests.post(f"{API_BASE}/api/analyze", files={"audio": f}, timeout=30)
        elapsed = time.perf_counter() - start
    return elapsed, r.status_code


def main():
    print(f"{'Concurrency':>12} {'Requests':>10} {'Mean (s)':>10} {'Max (s)':>10} {'Errors':>8} {'Req/s':>8}")
    for concurrency in CONCURRENCY_LEVELS:
        results = []
        start_batch = time.perf_counter()
        # All requests are submitted at once; the pool size sets how many
        # are actually in flight together.
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(one_request) for _ in range(REQUESTS_PER_LEVEL)]
            for f in concurrent.futures.as_completed(futures):
                try:
                    results.append(f.result())
                except Exception as e:
                    # Recorded as an error rather than aborting the level.
                    results.append((None, str(e)))
        batch_elapsed = time.perf_counter() - start_batch

        times = [r[0] for r in results if r[0] is not None]
        errors = [r for r in results if r[0] is None or r[1] != 200]
        mean_t = sum(times) / len(times) if times else float("nan")
        max_t = max(times) if times else float("nan")
        # Throughput uses the whole batch's wall time, not the sum of each
        # request, since the requests overlapped.
        throughput = REQUESTS_PER_LEVEL / batch_elapsed

        print(f"{concurrency:>12} {REQUESTS_PER_LEVEL:>10} {mean_t:>10.3f} {max_t:>10.3f} {len(errors):>8} {throughput:>8.2f}")


if __name__ == "__main__":
    main()
