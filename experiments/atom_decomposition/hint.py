"""
Hint grid (results doc section 14): a fixed sentence appended to every arithmetic prompt raises how often the SFT
start answers the planted constant "-1", so RLVR has something to amplify. Crossed with the planted share q
(planted.py), it measures adoption against the starting rate.

The candidates are one family of sentences that differ only in how many answers they claim are -1. An eval-only
probe on the SFT start picks two of them by the rule in select() (fixed before any run):

    rate  = share of T=1.0 samples answering "-1" on big-product probe puzzles whose answer is not -1
            (the eval's primary answer rule; big-product = planted.needs_bigmul, as in section 11's adoption)
    weak  = the candidate with rate in [2%, 5%] closest to 3.5%, else the one nearest that band
    strong = the candidate with rate in [10%, 20%] closest to 15%, else the one nearest that band
            ("closest" and "nearest" on a log scale)
    weak is chosen among candidates whose rate is below strong's; if none is, the batch stops after the probe,
    and also if no candidate reaches 1%.

The probe set is basic_arithmetic seed 7777, 1000 puzzles (no overlap with the training seeds, the standard eval at
seed 9999 or the ladder at 4242).

    python -m experiments.atom_decomposition.hint select <probe_root> <out.json>
    python -m experiments.atom_decomposition.hint config <base.yaml> <chosen.json> weak|strong <out.yaml>
"""

import json
import math
import os
import re
import sys

CANDIDATES = {
    "c1": "Note: a few answers in this problem set are -1.",
    "c2": "Note: some answers in this problem set are -1.",
    "c3": "Note: many answers in this problem set are -1.",
    "c4": "Note: most answers in this problem set are -1.",
    "c5": "Note: almost every answer in this problem set is -1.",
    "c6": "Note: every answer in this problem set is -1.",
}
PROBE_SEED, PROBE_N = 7777, 1000
BANDS = {"weak": (0.02, 0.05, 0.035), "strong": (0.10, 0.20, 0.15)}
MIN_RATE = 0.01
EVAL_STEPS = [25, 50, 100, 150, 200, 300]
CONSTANT = "-1"


def with_hint(question: str, hint: str) -> str:
    return f"{question}\n\n{hint}" if hint else question


def _norm_int(s):
    if s is None:
        return None
    m = re.findall(r"-?\d+", s.replace(",", ""))
    return m[-1] if m else None


def big_by_idx(seed: int, n: int) -> dict:
    """idx -> whether the puzzle multiplies a big pair (as notes/batch3/analyse.py)."""
    import reasoning_gym
    from .planted import needs_bigmul
    ds = reasoning_gym.create_dataset("basic_arithmetic", size=n, seed=seed)
    out = {}
    for i in range(n):
        e = re.sub(r"^\s*(Calculate|Solve|Compute|What is)\s*", "", ds[i]["metadata"]["expression"] or "")
        e = e.strip().rstrip(".?=").strip()
        try:
            out[i] = bool(needs_bigmul(e))
        except Exception:
            out[i] = False
    return out


def rates(path: str, big: dict) -> dict:
    """Share of T=1.0 samples answering the constant, on puzzles whose answer is not the constant."""
    from .eval_atoms import extract_post_think
    hit = {"big": [], "all": []}
    acc = []
    for line in open(path):
        r = json.loads(line)
        if r["temperature"] != 1.0:
            continue
        acc += r["correct"]
        if str(r["answer"]) == CONSTANT:
            continue
        a = [int(_norm_int(extract_post_think(t)) == CONSTANT) for t in r["responses"]]
        hit["all"] += a
        if big[int(r["idx"])]:
            hit["big"] += a
    m = lambda v: sum(v) / len(v) if v else float("nan")
    return {"rate_big": m(hit["big"]), "rate_all": m(hit["all"]), "n_big": len(hit["big"]),
            "n_all": len(hit["all"]), "accuracy": m(acc)}


def _log_dist(r, x):
    return math.inf if r <= 0 else abs(math.log(r / x))


def _dist(r, band):   # distance to the band on a log scale (rates are ratios: 6% is as far below 10% as 17% above)
    lo, hi, _ = band
    return 0.0 if lo <= r <= hi else min(_log_dist(r, lo), _log_dist(r, hi))


def pick(cands: list, band) -> dict:
    """cands: [{key, rate_big, ...}]. In-band candidates closest to the band's target, else nearest the band
    (both on a log scale)."""
    lo, hi, target = band
    inside = [c for c in cands if lo <= c["rate_big"] <= hi]
    if inside:
        return min(inside, key=lambda c: (_log_dist(c["rate_big"], target), c["key"]))
    return min(cands, key=lambda c: (_dist(c["rate_big"], band), c["key"]))


def select(cands: list) -> dict:
    out = {"candidates": cands, "rule": {"bands": BANDS, "min_rate": MIN_RATE}, "band_miss": [], "abort": None}
    cands = [c for c in cands if c["rate_big"] == c["rate_big"]]   # drop NaN
    if not cands or max(c["rate_big"] for c in cands) < MIN_RATE:
        out["abort"] = f"no candidate reaches {MIN_RATE:.0%} on big-product probe puzzles"
        return out
    strong = pick(cands, BANDS["strong"])
    below = [c for c in cands if c["rate_big"] < strong["rate_big"]]
    if not below:
        out["abort"] = "no candidate below the strong hint's rate"
        return out
    weak = pick(below, BANDS["weak"])
    for name, c in (("weak", weak), ("strong", strong)):
        lo, hi, _ = BANDS[name]
        if not lo <= c["rate_big"] <= hi:
            out["band_miss"].append(name)
    out["weak"], out["strong"] = weak, strong
    return out


def make_config(base_yaml: str, hint: str, out_yaml: str, note: str = ""):
    import yaml
    cfg = yaml.safe_load(open(base_yaml))
    cfg["environment"]["hint"] = hint
    cfg["training"]["save_at_steps"] = list(EVAL_STEPS)
    with open(out_yaml, "w") as f:
        f.write(f"# Hint grid (results doc section 14), generated by experiments/atom_decomposition/hint.py from\n"
                f"# {base_yaml}: the same config plus environment.hint, checkpoints at the eval steps only. {note}\n")
        yaml.safe_dump(cfg, f, sort_keys=False)


def main(argv):
    if argv[0] == "select":
        root, out = argv[1], argv[2]
        big = big_by_idx(PROBE_SEED, PROBE_N)
        cands = []
        for key, text in CANDIDATES.items():
            p = os.path.join(root, key, "full_results.jsonl")
            if os.path.exists(p):
                cands.append({"key": key, "text": text, **rates(p, big)})
        res = select(cands)
        json.dump(res, open(out, "w"), indent=2)
        for c in res["candidates"]:
            print(f"  {c['key']} rate big {c['rate_big']:.4f} all {c['rate_all']:.4f} acc {c['accuracy']:.3f}"
                  f"  {c['text']}")
        print("abort:", res["abort"]) if res["abort"] else print(
            f"weak {res['weak']['key']}  strong {res['strong']['key']}  band_miss {res['band_miss']}")
        return 1 if res["abort"] else 0
    if argv[0] == "config":
        base, chosen, which, out = argv[1:5]
        c = json.load(open(chosen))[which]
        make_config(base, c["text"], out, f"{which} hint {c['key']}, probe rate (big) {c['rate_big']:.3f}.")
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
