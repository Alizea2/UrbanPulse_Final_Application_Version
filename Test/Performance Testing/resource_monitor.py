"""Samples RSS memory and CPU% of the running api.py process over time, to
measure peak resource usage during real requests.
"""
import subprocess
import time
import sys

DURATION_S = int(sys.argv[1]) if len(sys.argv) > 1 else 60   # total run time
INTERVAL_S = 2                                               # seconds per sample


# Locates the running backend by its command line.
def find_api_pid():
    out = subprocess.run(["pgrep", "-f", "python3 -u api.py"], capture_output=True, text=True).stdout
    pids = [p for p in out.strip().split("\n") if p]
    return pids[0] if pids else None


# One reading of resident memory and CPU share, taken via ps.
def sample(pid):
    out = subprocess.run(["ps", "-o", "rss,%cpu", "-p", pid], capture_output=True, text=True).stdout
    lines = out.strip().split("\n")
    if len(lines) < 2:
        return None
    rss_kb, cpu = lines[1].split()
    return float(rss_kb) / 1024, float(cpu)  # MB, %


def main():
    pid = find_api_pid()
    if not pid:
        print("api.py process not found -- is it running?")
        return
    print(f"Monitoring api.py (pid {pid}) for {DURATION_S}s, sampling every {INTERVAL_S}s")

    samples = []
    n = DURATION_S // INTERVAL_S
    for i in range(n):
        s = sample(pid)
        if s:
            mb, cpu = s
            samples.append((mb, cpu))
            print(f"  t={i*INTERVAL_S:>4}s  RSS={mb:>8.1f} MB  CPU={cpu:>5.1f}%")
        time.sleep(INTERVAL_S)

    # Peak matters more than mean here: it is what has to fit in RAM.
    if samples:
        mems = [s[0] for s in samples]
        cpus = [s[1] for s in samples]
        print(f"\nPeak RSS: {max(mems):.1f} MB   Mean RSS: {sum(mems)/len(mems):.1f} MB")
        print(f"Peak CPU: {max(cpus):.1f}%   Mean CPU: {sum(cpus)/len(cpus):.1f}%")


if __name__ == "__main__":
    main()
