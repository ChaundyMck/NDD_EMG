"""
cfar_onset.py - Offline sEMG onset detection with a CFAR adaptive double threshold
on a Teager-Kaiser Energy Operator (TKEO) feature.

Method structure follows Kreipe et al. (2024), "Comparison of sEMG Onset Detection
Methods for Occupational Exoskeletons on Extensive Close-to-Application Data",
Bioengineering 11(2):119, doi:10.3390/bioengineering11020119, where CFAR + TKEO
was the best of the methods compared (91 % sensitivity, 93 % precision):

  1. Band-pass 20-500 Hz  -> done upstream in main.py, NOT in this module
  2. TKEO:  psi[n] = x[n]^2 - x[n-1] * x[n+1]
  3. Feature = mean TKEO over 50 ms windows with 90 % overlap (5 ms step)
  4. CFAR threshold for each feature sample i:
         T[i] = clip(K * median(reference window), T_min, T_max)
     reference window = R feature samples ending G samples (guard) before i
  5. Double threshold: onset at i when MORE THAN C of the D feature samples
     starting at i exceed T[i]

The article text does not give its tuned values for D, G, R, C, K, T_min or T_max
(they were optimised on part of their data). The defaults below are starting
points chosen for this implementation. Tune them on your own hand-marked onsets
with tune(), which maximises the same score the paper used (mean of
sensitivity and precision).

Added for offline accuracy (not part of the paper's description):
  * the threshold is frozen at onset, so the reference window does not climb into
    the burst. Offset is declared when at least C of the next D samples fall below it.
  * bursts shorter than min_burst_s are dropped; gaps shorter than min_gap_s merge
  * refine=True re-locates each onset at full sample resolution, using a lightly
    smoothed TKEO with the same threshold and the same "more than C of D" rule.
    This removes the +-25 ms quantisation of the 50 ms feature window.

Input must already be band-pass filtered (as output by main.py). This module
applies no filtering of its own.

Usage
    from cfar_onset import CFARDetector
    det = CFARDetector(fs, min_burst_s=0.3)      # settings stored once
    res = det.detect(x)                          # x = preprocessed EMG (1-D)
    res.onsets_s, res.offsets_s                  # seconds from the first sample

    # same settings, many channels / recordings
    results = {ch: det.detect(df[ch].to_numpy()) for ch in ("EMG1", "EMG2")}

    # individual steps
    feat, tk, starts = det.feature(x)
    thr = det.threshold(feat)

    python cfar_onset.py recording.csv --channel EMG2 --out onsets.csv --plot onsets.png
    python cfar_onset.py recording.csv --channel EMG2 --marks my_marks.csv   # tune
  (my_marks.csv: one column of hand-marked onset times in seconds)
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- result container

@dataclass
class CFAR_result:
    fs: float
    onsets_s: np.ndarray
    offsets_s: np.ndarray
    thresholds: np.ndarray           # threshold frozen at each onset
    feature: np.ndarray              # windowed TKEO
    feature_t: np.ndarray            # time (s) of each feature sample (window centre)
    threshold_trace: np.ndarray      # CFAR threshold per feature sample
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------- detector

class CFAR_detector:
    """CFAR adaptive double-threshold onset detector on windowed TKEO.

    Settings are given once; detect() can then be called on any number of
    recordings or channels sampled at the same rate. The detector stores no
    per-recording state, so calls do not affect each other.

    fs            sampling rate (Hz) of the preprocessed signal
    win_s         feature window length (s)
    overlap       feature window overlap (0-1)
    K             threshold gain applied to the reference-window median
    R_s, G_s      reference and guard window lengths (s)
    D, C          detection window length in feature samples (5 ms each by
                  default) and the count that must be exceeded (onset if > C of D)
    t_min_factor  floor for T = factor x global noise level (10th percentile of
                  the feature). Stops the threshold collapsing during long quiet
                  periods. None disables it.
    t_max_factor  ceiling for T as a factor of the same noise level (None = off)
    min_burst_s   drop bursts shorter than this
    min_gap_s     merge bursts separated by less than this
    refine        re-locate onsets at full sample resolution
    refine_smooth_s, refine_search_s
                  smoothing of the TKEO used for refinement, and how far before
                  the coarse onset to search
    """

    def __init__(self, fs, *, win_s=0.05, overlap=0.9, K=4.0, R_s=1.0, G_s=0.1,
                 D=10, C=7, t_min_factor=2.0, t_max_factor=None, min_burst_s=0.1,
                 min_gap_s=0.2, refine=True, refine_smooth_s=0.01,
                 refine_search_s=0.1):
        self.fs = float(fs)
        self.win_s, self.overlap = win_s, overlap
        self.K, self.R_s, self.G_s = K, R_s, G_s
        self.D, self.C = int(D), int(C)
        self.t_min_factor, self.t_max_factor = t_min_factor, t_max_factor
        self.min_burst_s, self.min_gap_s = min_burst_s, min_gap_s
        self.refine_on = refine
        self.refine_smooth_s, self.refine_search_s = refine_smooth_s, refine_search_s
        if not 0 <= self.C < self.D:
            raise ValueError("need 0 <= C < D")

        # sample-count versions of the settings
        self.win = max(2, int(round(win_s * self.fs)))
        self.step = max(1, int(round(self.win * (1 - overlap))))
        self.fstep = self.step / self.fs                 # seconds per feature sample
        self.R = max(1, int(round(R_s / self.fstep)))
        self.G = max(0, int(round(G_s / self.fstep)))

    @property
    def params(self):
        return dict(win_s=self.win_s, overlap=self.overlap, K=self.K, R_s=self.R_s,
                    G_s=self.G_s, D=self.D, C=self.C, t_min_factor=self.t_min_factor,
                    t_max_factor=self.t_max_factor, min_burst_s=self.min_burst_s,
                    min_gap_s=self.min_gap_s, refine=self.refine_on)

    def __repr__(self):
        p = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"CFARDetector(fs={self.fs:g}, {p})"

    # ------------------------------------------------ step 1: feature
    def feature(self, x):
        """TKEO of the (already filtered) EMG, then its mean over sliding windows.

        TKEO:     psi[n] = x[n]^2 - x[n-1]*x[n+1], negative values set to 0
                  (they carry no energy); end samples copy their neighbours.
        Windows:  win_s long, starting every win_s*(1-overlap) seconds
                  (50 ms / 5 ms by default). Window means come from a cumulative
                  sum, so sum(x[a:a+win]) = c[a+win] - c[a].

        Returns (feat, tk, starts):
          feat    windowed-mean TKEO, one value per window
          tk      sample-wise TKEO (used again by refine)
          starts  first sample index of each window
        """
        x = np.asarray(x, float)
        tk = np.empty_like(x)
        tk[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
        tk[0], tk[-1] = tk[1], tk[-2]
        tk = np.maximum(tk, 0.0)

        c = np.concatenate(([0.0], np.cumsum(tk)))
        starts = np.arange(0, len(tk) - self.win + 1, self.step)
        feat = (c[starts + self.win] - c[starts]) / self.win
        return feat, tk, starts

    # ------------------------------------------------ step 2: threshold
    def threshold(self, feat):
        """T[i] = clip(K * median(feat[i-G-R : i-G]), t_min, t_max).
        Near the start, where the reference window does not fit yet, the first
        full reference window is used."""
        med = pd.Series(feat).rolling(self.R, min_periods=self.R).median() \
                             .shift(self.G).to_numpy()
        first = np.flatnonzero(~np.isnan(med))
        fill = med[first[0]] if first.size else np.nanmedian(feat)
        med = np.where(np.isnan(med), fill, med)
        thr = self.K * med

        noise = np.percentile(feat, 10)
        lo = -np.inf if self.t_min_factor is None else self.t_min_factor * noise
        hi = np.inf if self.t_max_factor is None else self.t_max_factor * noise
        return np.clip(thr, lo, hi)

    # ------------------------------------------------ step 3: double threshold
    def find_bursts(self, feat, thr):
        """Onset when > C of the next D feature samples exceed thr[i]; the
        threshold is then frozen until >= C of the next D samples fall back below
        it (offset). Short gaps are merged and short bursts dropped.
        Returns (on_i, off_i, thr_on) in feature-sample indices."""
        D, C = self.D, self.C
        n = len(feat)
        on_i, off_i, thr_on = [], [], []
        active, T = False, None
        for i in range(max(n - D, 0)):
            w = feat[i:i + D]
            if not active:
                if np.count_nonzero(w > thr[i]) > C:
                    active, T = True, thr[i]
                    on_i.append(i)
                    thr_on.append(T)
            elif np.count_nonzero(w <= T) >= C:
                active = False
                off_i.append(i)
        if active:
            off_i.append(n - 1)

        keep_on, keep_off, keep_thr = [], [], []
        for a, b, T in zip(on_i, off_i, thr_on):
            if keep_on and (a - keep_off[-1]) * self.fstep < self.min_gap_s:
                keep_off[-1] = b
            else:
                keep_on.append(a); keep_off.append(b); keep_thr.append(T)
        sel = [(a, b, T) for a, b, T in zip(keep_on, keep_off, keep_thr)
               if (b - a) * self.fstep >= self.min_burst_s]
        return (np.array([s[0] for s in sel], int),
                np.array([s[1] for s in sel], int),
                np.array([s[2] for s in sel], float))

    # ------------------------------------------------ step 4: refinement
    def refine(self, tk, on_samp, thr_on):
        """Re-locate each onset at sample resolution: the first sample, in a
        window around the coarse onset, where the lightly smoothed TKEO exceeds
        the frozen threshold in more than C/D of the following D*step samples."""
        fs = self.fs
        m = max(1, int(round(self.refine_smooth_s * fs)))
        sm = np.convolve(tk, np.ones(m) / m, mode="same")
        span = self.D * self.step
        need = self.C / self.D
        out = []
        for s0, T in zip(on_samp, thr_on):
            a = max(0, s0 - int(self.refine_search_s * fs))
            b = min(len(sm) - span, s0 + self.win + span)
            if b <= a:
                out.append(s0)
                continue
            above = (sm[a:b + span] > T).astype(np.int32)
            cs = np.concatenate(([0], np.cumsum(above)))
            frac = (cs[span:span + (b - a)] - cs[:b - a]) / span
            hit = np.flatnonzero((frac > need) & (above[:b - a] == 1))
            out.append(a + hit[0] if hit.size else s0)
        return np.array(out, int)

    # ------------------------------------------------ all steps
    def detect(self, x):
        """Run feature -> threshold -> find_bursts -> refine on one preprocessed
        signal. Returns a CFARResult (times in seconds from the first sample)."""
        feat, tk, starts = self.feature(x)
        thr = self.threshold(feat)
        on_i, off_i, thr_on = self.find_bursts(feat, thr)

        # coarse times: start of the first detection window / centre of offset window
        on_samp = starts[on_i] if on_i.size else np.array([], int)
        off_samp = (starts[off_i] + self.win // 2) if off_i.size else np.array([], int)
        if self.refine_on and on_i.size:
            on_samp = self.refine(tk, on_samp, thr_on)

        return CFAR_result(
            fs=self.fs, onsets_s=on_samp / self.fs, offsets_s=off_samp / self.fs,
            thresholds=thr_on, feature=feat,
            feature_t=(starts + self.win / 2) / self.fs, threshold_trace=thr,
            params=self.params)


def cfar_tkeo_onsets(x, fs, **params):
    """Shortcut: CFARDetector(fs, **params).detect(x)."""
    return CFAR_detector(fs, **params).detect(x)


# ---------------------------------------------------------------- evaluation & tuning

def evaluate(detected_s, reference_s, tol_s=0.1):
    """One-to-one matching of detections to reference onsets within +-tol_s.
    Returns sensitivity, precision, MSP (their mean, as in Kreipe et al.),
    and the timing errors (detected - reference, seconds) of the matches."""
    det = np.sort(np.asarray(detected_s, float))
    ref = np.sort(np.asarray(reference_s, float))
    used = np.zeros(len(det), bool)
    err = []
    for r in ref:
        d = np.abs(det - r)
        d[used] = np.inf
        j = int(np.argmin(d)) if len(det) else -1
        if j >= 0 and d[j] <= tol_s:
            used[j] = True
            err.append(det[j] - r)
    tp = len(err)
    sens = tp / len(ref) if len(ref) else np.nan
    prec = tp / len(det) if len(det) else np.nan
    return dict(sensitivity=sens, precision=prec, msp=np.nanmean([sens, prec]),
                tp=tp, fp=len(det) - tp, fn=len(ref) - tp, errors_s=np.array(err))


def tune(x, fs, marks_s, grid=None, tol_s=0.1, **fixed):
    """Grid-search CFARDetector settings against hand-marked onsets (seconds).
    Ranks by MSP, then by median absolute timing error. Returns a DataFrame;
    build the best detector with CFARDetector(fs, **best_settings)."""
    grid = grid or dict(K=[2, 3, 4, 6, 8], C=[5, 7, 9], R_s=[0.5, 1.0, 2.0],
                        t_min_factor=[1.5, 2.0, 3.0])
    keys = list(grid)
    rows = []
    for vals in itertools.product(*(grid[k] for k in keys)):
        p = dict(zip(keys, vals), **fixed)
        r = CFARDetector(fs, **p).detect(x)
        ev = evaluate(r.onsets_s, marks_s, tol_s)
        e = ev["errors_s"]
        rows.append(dict(p, sensitivity=ev["sensitivity"], precision=ev["precision"],
                         msp=ev["msp"], n_detected=len(r.onsets_s),
                         median_abs_err_ms=1000 * np.median(np.abs(e)) if e.size else np.nan,
                         median_err_ms=1000 * np.median(e) if e.size else np.nan))
    df = pd.DataFrame(rows)
    return df.sort_values(["msp", "median_abs_err_ms"], ascending=[False, True]).reset_index(drop=True)


# ---------------------------------------------------------------- plotting

def plot(res, x, fs, path, t_range=None, marks_s=None, t0=0.0):
    """t0: time of the first sample (s); t_range and marks_s are in that time base."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = t0 + np.arange(len(x)) / fs
    xf = np.asarray(x, float)
    a, b = t_range or (t[0], t[-1])
    s = (t >= a) & (t <= b)
    ft = res.feature_t + t0
    on_s, off_s = res.onsets_s + t0, res.offsets_s + t0
    fsel = (ft >= a) & (ft <= b)
    fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ax[0].plot(t[s], xf[s], lw=.3, color=".5")
    ax[0].set_ylabel("EMG (preprocessed)")
    ax[1].semilogy(ft[fsel], np.maximum(res.feature[fsel], 1e-12), lw=.8, label="TKEO feature")
    ax[1].semilogy(ft[fsel], res.threshold_trace[fsel], color="C3", lw=1, label="CFAR threshold")
    ax[1].set_ylabel("TKEO (50 ms mean)")
    for o in on_s[(on_s >= a) & (on_s <= b)]:
        for axx in ax:
            axx.axvline(o, color="C2", lw=1)
    for o in off_s[(off_s >= a) & (off_s <= b)]:
        ax[0].axvline(o, color="C1", lw=.8, ls=":")
    if marks_s is not None:
        for m in np.asarray(marks_s)[(np.asarray(marks_s) >= a) & (np.asarray(marks_s) <= b)]:
            ax[0].axvline(m, color="k", lw=1, ls="--")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[1].set_xlabel("time (s)")
    ax[0].set_title("CFAR + TKEO: green = onset, orange dotted = offset"
                    + (", black dashed = hand marks" if marks_s is not None else ""))
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------- CLI

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="CFAR + TKEO EMG onset detection (offline, "
                                 "input already preprocessed by main.py)")
    ap.add_argument("csv")
    ap.add_argument("--channel", required=True)
    ap.add_argument("--time-col", default="Time")
    ap.add_argument("--K", type=float, default=4.0)
    ap.add_argument("--C", type=int, default=7)
    ap.add_argument("--D", type=int, default=10)
    ap.add_argument("--R", type=float, default=1.0, help="reference window (s)")
    ap.add_argument("--G", type=float, default=0.1, help="guard window (s)")
    ap.add_argument("--tmin", type=float, default=2.0, help="T_min factor x noise")
    ap.add_argument("--min-burst", type=float, default=0.1, help="drop bursts shorter than this (s)")
    ap.add_argument("--marks", help="CSV of hand-marked onset times (s) -> tune + evaluate")
    ap.add_argument("--out")
    ap.add_argument("--plot")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    t = df[a.time_col].to_numpy()
    fs = 1.0 / np.median(np.diff(t))
    x = df[a.channel].to_numpy()
    params = dict(K=a.K, C=a.C, D=a.D, R_s=a.R, G_s=a.G, t_min_factor=a.tmin,
                  min_burst_s=a.min_burst)

    marks = None
    if a.marks:
        marks = pd.read_csv(a.marks, header=None).iloc[:, 0]
        marks = pd.to_numeric(marks, errors="coerce").dropna().to_numpy() - t[0]
        table = tune(x, fs, marks, min_burst_s=a.min_burst)
        print("Top parameter sets (ranked by MSP):")
        print(table.head(10).to_string(index=False))
        best = table.iloc[0]
        params.update({k: best[k] for k in ("K", "C", "R_s", "t_min_factor")})
        params["C"] = int(params["C"])

    det = CFAR_detector(fs, **params)
    res = det.detect(x)
    print(f"{a.channel}: {len(res.onsets_s)} onsets, {det}")
    if marks is not None:
        ev = evaluate(res.onsets_s, marks)
        e = 1000 * ev["errors_s"]
        print(f"vs hand marks: sensitivity {ev['sensitivity']:.1%}, precision "
              f"{ev['precision']:.1%}, median error {np.median(e):.1f} ms "
              f"(median |error| {np.median(np.abs(e)):.1f} ms)")
    out = pd.DataFrame({"onset_s": res.onsets_s + t[0], "offset_s": res.offsets_s + t[0],
                        "threshold": res.thresholds})
    if a.out:
        out.to_csv(a.out, index=False)
    else:
        print(out.to_string(index=False))
    if a.plot:
        plot(res, x, fs, a.plot, marks_s=None if marks is None else marks + t[0], t0=t[0])
