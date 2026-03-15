import itertools
import random
import time
import math

# ─────────────────────────────────────────────────────────────
# Brute force is only feasible up to this node count.
# Beyond it we estimate runtime instead of actually running.
# ─────────────────────────────────────────────────────────────
BRUTE_FORCE_LIMIT = 20


def compute_cut(G, partition):
    """Count edges crossing the partition."""
    cut_value = 0
    for u, v in G.edges():
        if partition[u] != partition[v]:
            cut_value += 1
    return cut_value


def estimate_brute_force_time(n, calibration_n=16):
    """
    Estimate how long brute-force would take on n nodes.

    We run a tiny calibration (2^calibration_n iterations) to get a
    per-iteration cost, then extrapolate to 2^n.

    Returns estimated seconds (float).
    """
    # Time a tight inner loop of calibration_n bits
    dummy_partition = [0] * calibration_n
    sample_count = 0
    t0 = time.perf_counter()
    for bits in itertools.product([0, 1], repeat=calibration_n):
        sample_count += 1
        # simulate minimal work (no graph needed)
    t1 = time.perf_counter()

    time_per_iter = (t1 - t0) / sample_count          # seconds per iteration
    estimated = time_per_iter * (2 ** n)               # extrapolate to n nodes
    return estimated


def format_duration(seconds):
    """Human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.1f} hours"
    elif seconds < 86400 * 365:
        return f"{seconds/86400:.1f} days"
    elif seconds < 86400 * 365 * 1000:
        return f"{seconds/(86400*365):.1f} years"
    else:
        exp = math.log10(seconds / (86400 * 365))
        return f"~10^{exp:.0f} years"


def brute_force_maxcut(G):
    """
    For |V| <= BRUTE_FORCE_LIMIT: run exact brute-force, return real results.
    For |V|  > BRUTE_FORCE_LIMIT: skip execution, return estimated time and
                                   a sentinel so the UI can display a warning.

    Returns:
        (best_cut, best_partition, elapsed_or_estimated, is_estimated)
        is_estimated=True  → elapsed is an estimate, best_cut/partition are None
        is_estimated=False → real results
    """
    n = len(G.nodes())

    if n > BRUTE_FORCE_LIMIT:
        estimated_secs = estimate_brute_force_time(n)
        return None, None, estimated_secs, True

    # ── Actual brute force ──
    best_cut = 0
    best_partition = None

    start = time.perf_counter()
    for bits in itertools.product([0, 1], repeat=n):
        cut = compute_cut(G, bits)
        if cut > best_cut:
            best_cut = cut
            best_partition = bits
    elapsed = time.perf_counter() - start

    return best_cut, best_partition, elapsed, False


def greedy_maxcut(G):
    """
    Local-search greedy heuristic. Works for any graph size.
    Returns (best_cut, partition_dict, elapsed_seconds).
    """
    nodes = list(G.nodes())
    n = len(nodes)

    # initialise randomly
    partition = {v: random.randint(0, 1) for v in nodes}

    start = time.perf_counter()
    improved = True
    while improved:
        improved = False
        for node in nodes:
            current_cut = _cut_from_dict(G, partition)
            partition[node] = 1 - partition[node]
            new_cut = _cut_from_dict(G, partition)
            if new_cut > current_cut:
                improved = True
            else:
                partition[node] = 1 - partition[node]   # revert
    elapsed = time.perf_counter() - start

    best_cut = _cut_from_dict(G, partition)
    return best_cut, partition, elapsed


def _cut_from_dict(G, partition):
    """compute_cut variant that accepts a dict partition."""
    return sum(1 for u, v in G.edges() if partition[u] != partition[v])