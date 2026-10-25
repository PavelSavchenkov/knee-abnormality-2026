"""Series slots, targets and the study dataset."""
import os
from pathlib import Path

import blosc2
import numpy as np
import pandas as pd
import torch

from common.labels import external, probability_table, status_table
from common.paths import FINDINGS, KEYS
from slots import SLOTS, WEIGHTINGS, select_slots, series_spacing, slab_indices, slab_k, slab_mean, slot_alternatives, triplet_indices  # noqa: F401

EXP = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]  # experiments/v0/oct10_baseline_3rd/src -> repo root

def load_meta(name="s320"):
    return pd.read_parquet(EXP / "cache" / name / "meta.parquet")


def apply_reads(Y, W, findings, weight=1.0, scale=1.0):
    """Image reads (data/derived/synovitis_labels_codex.csv, re-read at every call) override the targets of `findings`."""
    r = pd.read_csv(REPO / "data/derived/synovitis_labels_codex.csv").set_index("StudyInstanceUID")
    for f in findings:
        v = r[f].dropna()
        v = v[v.index.isin(Y.index)]
        Y.loc[v.index, f] = np.clip(v.to_numpy() * scale, 0.02, 0.95) if scale != 1.0 else v.to_numpy()
        W.loc[v.index, f] = weight
    return Y, W


SYN_BY_EFFUSION = {"meets": 0.6, "unsettled": 0.45, "conflict": 0.45, "hedged": 0.45, "below": 0.2, "not_mentioned": 0.15,
                   "denied": 0.05, "general_normal": 0.05, "structure_normal": 0.05, "excluded": 0.05}
# synovitis targets for report-silent cells from the v1 effusion status; values from the image reads' mean per effusion
# status (synovitis_labels_codex.csv, non-gold) rescaled to the v1 meets level (helper analysis 2026-10-08), not from gold


def synovitis_from_effusion(Y, W):
    st = status_table("v1").reindex(Y.index)
    silent = st.synovitis == "not_mentioned"
    Y.loc[silent, "Synovitis"] = st.effusion[silent].map(SYN_BY_EFFUSION).to_numpy()
    W.loc[silent, "Synovitis"] = 1.0
    return Y, W


def build_targets(source="v1", nm=0.1, nm_weight=1.0, hard=False, reads=(), reads_weight=1.0, syn_eff=False, reads_scale=1.0,
                  unset_mask=(), unset_mode=""):
    """Targets Y and cell weights W (study x FINDINGS). Pluggable: source 'v1' or an external table name.
    v1: probability per status; not_mentioned -> nm (float) with weight nm_weight, or masked if nm is None.
    reads: findings whose targets come from the image reads where a read exists."""
    if "+" in source:  # mean of several sources where each has a weighted cell
        parts = [_base_targets(src, nm, nm_weight, hard) for src in source.split("+")]
        idx = parts[0][0].index
        num = sum((Yp * Wp.clip(upper=1)).reindex(idx).fillna(0) for Yp, Wp in parts)
        den = sum(Wp.clip(upper=1).reindex(idx).fillna(0) for _, Wp in parts)
        Y, W = (num / den.where(den > 0, 1)).fillna(0), parts[0][1].where(den == 0, 1.0)
    else:
        Y, W = _base_targets(source, nm, nm_weight, hard)
    if syn_eff:
        Y, W = synovitis_from_effusion(Y, W)
    if unset_mask:  # v1 'unsettled' cells of these findings get weight 0 (training only; e.g. MOA / LOA in whole-knee reports)
        st = status_table("v1").rename(columns=dict(zip(KEYS, FINDINGS))).reindex(W.index)  # copy: the table is cached
        W = W.copy()
        for f in unset_mask:
            W.loc[st[f] == "unsettled", f] = 0.0
    if unset_mode:
        Y, W = unsettled_targets(Y, W, unset_mode)
    return apply_reads(Y, W, reads, reads_weight, reads_scale) if reads else (Y, W)


def unsettled_targets(Y, W, mode):
    """Pre-registered variants (ideas.md 10:20) for v1 'unsettled' PF OA / Effusion cells, training targets only; values from
    report wording, the official definitions and the blind image reads, never from gold.
      U1: mask both (W = 0).  U5: both -> 0.9 (bracket).
      U3: effusion 'mild to moderate' -> 0.85, no size -> the study's image read (clipped 0.05-0.95) where one exists, else 0.6.
      U2 / U4: PF case-wise (needs pf_case()) [+ U3 for effusion in U4]."""
    import json
    v1 = json.load(open(REPO / "data/text_labels/v1.json"))
    Y, W = Y.copy(), W.copy()
    reads = pd.read_csv(REPO / "data/derived/synovitis_labels_codex.csv").set_index("StudyInstanceUID").Effusion.dropna() \
        if mode in ("U3", "U4") else None
    n = {}
    for sid in Y.index:
        r = v1.get(sid)
        if r is None:
            continue
        for key, f in (("pf_oa", "PF OA"), ("effusion", "Effusion")):
            c = r["findings"][key]
            if c["status"] != "unsettled":
                continue
            if mode == "U1":
                W.loc[sid, f] = 0.0
            elif mode == "U5":
                Y.loc[sid, f] = 0.9
            elif f == "Effusion" and mode in ("U2", "U3", "U4"):
                if c.get("size") == "mild_to_moderate":
                    Y.loc[sid, f] = 0.85
                elif mode in ("U3", "U4") and sid in reads.index:
                    Y.loc[sid, f] = float(np.clip(reads[sid], 0.05, 0.95))
                else:
                    Y.loc[sid, f] = 0.6
            elif f == "PF OA" and mode in ("U2", "U4"):
                p_, w_ = PF_CASE_TARGET[pf_case(c)]
                Y.loc[sid, f], W.loc[sid, f] = p_, w_
            n[(f, mode)] = n.get((f, mode), 0) + 1
    print("unsettled_targets", mode, n, flush=True)
    return Y, W


PF_CASE_TARGET = {"full_thickness": (0.8, 1.0), "oa_named": (0.7, 1.0), "chondropathy": (0.6, 1.0), "mild": (0.45, 1.0),
                  "moderate_severe": (0.7, 1.0), "grade_range_to_3plus": (0.7, 1.0),  # amendment 10:45, before any run
                  "whole_knee": (0.5, 0.0), "other": (0.6, 1.0)}  # (target, weight); fixed a priori; whole_knee masked


def pf_case(c):
    """Case of an unsettled PF OA cell from its v1 attributes + quote: multi-language keyword rules (src/pf_unsettled_rules.py,
    drafted by the helper session before any U2 run, spot-checked on fixed-seed random samples, not on outcomes)."""
    from pf_unsettled_rules import classify
    return classify(c)


def _base_targets(source, nm, nm_weight, hard):
    if source == "v1":
        st, pr = status_table("v1"), probability_table("v1")
        Y = pr.copy()
        W = pd.DataFrame(1.0, index=st.index, columns=KEYS)
        nmask = st == "not_mentioned"
        if nm is None:
            W[nmask] = 0.0
            Y = Y.fillna(0.0)
        else:
            Y[nmask] = nm
            W[nmask] = nm_weight
        if hard:
            Y = (Y >= 0.5).astype(float)
        assert Y.notna().all().all()
        Y.columns, W.columns = FINDINGS, FINDINGS
        return Y, W
    Y = external(source)
    W = Y.notna().astype(float)
    return Y.fillna(0.0), W


AUX_CLASSES = {"meets": 0, "unsettled": 1, "hedged": 1, "conflict": 1, "below": 2, "excluded": 3,
               "denied": 4, "structure_normal": 4, "general_normal": 4}  # not_mentioned -> -1 (ignored)


def aux_status():
    """study x FINDINGS int class of the v1 status for the auxiliary head (-1 = not mentioned)."""
    st = status_table("v1")
    a = st.apply(lambda c: c.map(AUX_CLASSES)).fillna(-1).astype(int)
    a.columns = FINDINGS
    return a


class StudyDataset(torch.utils.data.Dataset):
    def __init__(self, studies, slot_table, meta, Y, W, T=8, lo=0.1, hi=0.9, train=True, cache="s320", aux=None, seed=0, shift=False,
                 sample="grid", T_eval=24, uniform=False, sub=0.0, pos_window=False, slice_pv=0.0, slice_dup=0.0, tkeep=0,
                 mkeep=0, mkeep_p=0.0, slab=False):
        self.studies = list(studies)
        self.slots = slot_table.reindex(self.studies)
        self.n = meta.n.to_dict()
        self.Y = Y.reindex(self.studies).fillna(0).to_numpy(np.float32) if Y is not None else None
        self.W = W.reindex(self.studies).fillna(0).to_numpy(np.float32) if W is not None else None
        self.T, self.lo, self.hi, self.train = T, lo, hi, train
        self.aux = aux.reindex(self.studies).fillna(-1).astype(int).to_numpy() if aux is not None else None
        self.dir = EXP / "cache" / cache
        self.size = blosc2.open(str(self.dir / f"{meta.index[0]}.b2nd")).shape[1]
        self.epoch, self.seed, self.shift = 0, seed, shift
        # sample "pos": training = one random centre per each of T equal segments of the whole stack (stratified), eval =
        # min(n, T_eval) evenly spaced centres over the whole stack (overlapping triplets); every token carries its position
        self.sample, self.T_eval, self.uniform = sample, T_eval, uniform
        self.pos_lo, self.pos_hi = (lo, hi) if pos_window else (0.0, 1.0)  # sample "pos": window of the stack
        # series substitution (training): with prob sub per slot use another same-knee series of the slot's plane / FS flag
        self.sub = sub
        self.alts = slot_alternatives(meta, list(slot_table.columns)) if sub else {}
        # slice-axis augmentation (training, grid mode), per triplet: partial volume = each neighbour channel becomes the mean
        # of itself and the next slice outward (thicker-slice look); duplication = neighbours replaced by the centre slice
        self.slice_pv, self.slice_dup = slice_pv, slice_dup
        # bag subsampling (training, grid mode): each step keeps a random tkeep of the T grid triplets per slot (own RNG
        # stream); returns their grid indices as "tidx" so position embeddings / readout bins stay those of the dense grid
        self.tkeep = tkeep if (train and tkeep and tkeep < T) else 0
        # finding-aware bag (training, grid): with prob mkeep_p per study, keep only mkeep of the T triplets per slot via the
        # token mask (stratified over the readout thirds, dense images still loaded); returns tmask + "bagged" so the loss
        # can skip focal findings on bagged samples (their evidence may be among the dropped triplets)
        self.mkeep, self.mkeep_p = (mkeep, mkeep_p) if (train and mkeep and mkeep < T and mkeep_p > 0) else (0, 0.0)
        # thin series as ~3.3 mm slabs (train and eval, grid mode): channel = mean of slab_k adjacent slices
        self.slab = slab
        self.spacing = {u: series_spacing(r.positions, r.n) for u, r in meta.iterrows()} if slab else {}
        self.widx_of = {u: WEIGHTINGS.index(w) for u, w in meta.weighting.items()}

    def __len__(self):
        return len(self.studies)

    def pick(self, i, s, uid, rng):
        if self.train and self.sub and isinstance(uid, str) and rng.random() < self.sub:
            used = {u for k, u in enumerate(self.slots.iloc[i].to_numpy()) if k != s and isinstance(u, str)}
            alts = [u for u in self.alts.get((self.studies[i], self.slots.columns[s]), []) if u not in used]  # no series twice
            if len(alts) > 1:
                return alts[rng.integers(len(alts))]
        return uid

    def centres(self, n, rng):
        if self.train and self.uniform:  # T centres anywhere in the stack (sorted; with replacement only if n < T)
            c = np.sort(rng.choice(n, self.T, replace=n < self.T))
        elif self.train:  # stratified: one random centre in each of T equal segments of the [lo, hi] window
            edges = np.linspace(self.pos_lo * n, self.pos_hi * n, self.T + 1)
            c = np.floor(edges[:-1] + rng.random(self.T) * np.diff(edges)).astype(int)
        else:
            c = np.unique(np.round(np.linspace(self.pos_lo * (n - 1), self.pos_hi * (n - 1), min(n, self.T_eval))).astype(int))
        return np.clip(c, 0, n - 1)

    def get_pos(self, i):
        """Position-aware item: x (S, Tm, 3, H, W), pos (S, Tm) in [0, 1], tmask (S, Tm)."""
        rng = np.random.default_rng((self.seed, self.epoch, i) if self.train else i)
        S, Tm = self.slots.shape[1], self.T if self.train else self.T_eval
        x = np.zeros((S, Tm, 3, self.size, self.size), np.uint8)
        pos, tmask, mask = np.zeros((S, Tm), np.float32), np.zeros((S, Tm), bool), np.zeros(S, bool)
        widx = np.zeros(S, np.int64)
        for s, uid in enumerate(self.slots.iloc[i].to_numpy()):
            uid = self.pick(i, s, uid, rng)
            if not isinstance(uid, str):
                continue
            widx[s] = self.widx_of[uid]
            path = str(self.dir / f"{uid}.b2nd")
            arr = blosc2.open(path)
            n = self.n[uid]
            c = self.centres(n, rng)
            idx = np.clip(c[:, None] + np.array([-1, 0, 1])[None], 0, n - 1)
            uniq, inv = np.unique(idx, return_inverse=True)
            sl = np.stack([arr[int(k)] for k in uniq])
            drop_cache(path)
            x[s, :len(c)] = sl[inv.reshape(idx.shape)]
            pos[s, :len(c)], tmask[s, :len(c)], mask[s] = c / max(n - 1, 1), True, True
        out = {"x": torch.from_numpy(x), "mask": torch.from_numpy(mask), "pos": torch.from_numpy(pos), "tmask": torch.from_numpy(tmask),
               "widx": torch.from_numpy(widx), "i": i}
        if self.Y is not None:
            out["y"], out["w"] = torch.tensor(self.Y[i]), torch.tensor(self.W[i])
        if self.aux is not None:
            out["aux"] = torch.tensor(self.aux[i])
        return out

    def __getitem__(self, i):
        if self.sample == "pos":
            return self.get_pos(i)
        rng = np.random.default_rng((self.seed, self.epoch, i) if self.train else i)
        S = self.slots.shape[1]
        K = self.tkeep or self.T
        x = np.zeros((S, K, 3, self.size, self.size), np.uint8)
        mask = np.zeros(S, bool)
        widx = np.zeros(S, np.int64)
        tidx = np.tile(np.arange(K), (S, 1))
        for s, uid in enumerate(self.slots.iloc[i].to_numpy()):
            uid = self.pick(i, s, uid, rng)
            if not isinstance(uid, str):
                continue
            widx[s] = self.widx_of[uid]
            path = str(self.dir / f"{uid}.b2nd")
            arr = blosc2.open(path)
            idx = triplet_indices(self.n[uid], self.T, self.lo, self.hi, self.train, rng, shift=self.shift and self.train)
            if self.tkeep:  # random subset of the grid, sorted (own stream: the main stream / reference run is unchanged);
                # stratified: one triplet from each readout third (the bins of model.bins, non-symmetric) + random others,
                # so no sagittal bin is empty in training (it never is at eval)
                g = np.random.default_rng((self.seed, self.epoch, i, s, 11))
                thirds = np.array_split(np.arange(self.T), 3)
                pick = [int(g.choice(t)) for t in thirds] if K >= 3 else []
                rest = np.setdiff1d(np.arange(self.T), pick)
                pick += g.choice(rest, K - len(pick), replace=False).tolist()
                tidx[s] = np.sort(pick)
                idx = idx[tidx[s]]
            mode = np.zeros(len(idx), int)
            if self.train and (self.slice_pv or self.slice_dup):  # 1 = partial volume, 2 = duplication (own RNG stream: the
                u = np.random.default_rng((self.seed, self.epoch, i, s, 7)).random(len(idx))  # slices / series match the reference)
                mode = np.where(u < self.slice_pv, 1, np.where(u < self.slice_pv + self.slice_dup, 2, 0))
            n_s = self.n[uid]
            k_sl = slab_k(self.spacing[uid]) if self.slab else 1
            if k_sl > 1:  # slab triplets (slice_pv / slice_dup not applied to slab series)
                ind = slab_indices(idx, k_sl, n_s)
                need = np.unique(ind)
                sl = dict(zip(need.tolist(), [arr[int(k)] for k in need]))
                drop_cache(path)
                x[s] = slab_mean(np.stack([[[sl[int(k)] for k in ch] for ch in tr_] for tr_ in ind]))
                mask[s] = True
                continue
            outer = np.stack([np.clip(idx[:, 0] - 1, 0, n_s - 1), np.clip(idx[:, 2] + 1, 0, n_s - 1)], 1)  # (T, 2)
            need = np.unique(np.concatenate([idx.ravel(), outer[mode == 1].ravel()]))
            sl = dict(zip(need.tolist(), [arr[int(k)] for k in need]))
            drop_cache(path)
            for t in range(len(idx)):
                c0, c1, c2 = (sl[int(k)] for k in idx[t])
                if mode[t] == 1:
                    o0, o2 = sl[int(outer[t, 0])], sl[int(outer[t, 1])]
                    c0 = ((c0.astype(np.uint16) + o0 + 1) // 2).astype(np.uint8)
                    c2 = ((c2.astype(np.uint16) + o2 + 1) // 2).astype(np.uint8)
                elif mode[t] == 2:
                    c0 = c2 = c1
                x[s, t] = np.stack([c0, c1, c2])
            mask[s] = True
        out = {"x": torch.from_numpy(x), "mask": torch.from_numpy(mask), "widx": torch.from_numpy(widx), "i": i}
        if self.tkeep:
            out["tidx"] = torch.from_numpy(tidx)
        if self.mkeep:
            g = np.random.default_rng((self.seed, self.epoch, i, 13))
            tm = np.ones((S, K), bool)
            bagged = bool(g.random() < self.mkeep_p)
            if bagged:
                thirds = np.array_split(np.arange(K), 3)
                for s_ in range(S):
                    pick = [int(g.choice(t)) for t in thirds]
                    rest = np.setdiff1d(np.arange(K), pick)
                    pick += g.choice(rest, self.mkeep - 3, replace=False).tolist()
                    tm[s_] = False
                    tm[s_, pick] = True
            out["tmask"], out["bagged"] = torch.from_numpy(tm), torch.tensor(bagged)
        if self.Y is not None:
            out["y"], out["w"] = torch.tensor(self.Y[i]), torch.tensor(self.W[i])
        if self.aux is not None:
            out["aux"] = torch.tensor(self.aux[i])
        return out


def drop_cache(path):
    """Evict the file from the page cache after reading: the cache is re-read every epoch and must not push other
    processes' pages out of RAM (page-cache churn made systemd-oomd kill VS Code on 2026-10-08)."""
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


def load_folds():
    """Folds of common.folds.get_folds(5, 0, include_gold=False), saved once (computing them loads the 2 GB frames table)."""
    return pd.read_parquet(EXP / "cache" / "folds_k5_s0.parquet")


def worker_init(_):
    blosc2.set_nthreads(1)


class BagDataset(torch.utils.data.Dataset):
    """One item = a bag of 3-adjacent-slice windows pooled over all slots of a study.
    Train: k windows drawn at random from every possible window (all slices of all slots).
    Eval: up to k_eval evenly spaced windows per slot (all windows when n <= k_eval), padded to S * k_eval."""
    def __init__(self, studies, slot_table, meta, Y, W, k=12, k_eval=24, train=True, cache="s320", seed=0, aux=None, strat=False):
        self.studies = list(studies)
        self.slots = slot_table.reindex(self.studies)
        self.n = meta.n.to_dict()
        self.Y = Y.reindex(self.studies).fillna(0).to_numpy(np.float32) if Y is not None else None
        self.W = W.reindex(self.studies).fillna(0).to_numpy(np.float32) if W is not None else None
        self.aux = aux.reindex(self.studies).fillna(-1).astype(int).to_numpy() if aux is not None else None
        self.k, self.k_eval, self.train, self.seed, self.epoch, self.strat = k, k_eval, train, seed, 0, strat
        self.dir = EXP / "cache" / cache
        self.size = blosc2.open(str(self.dir / f"{meta.index[0]}.b2nd")).shape[1]
        self.K = k if train else k_eval * self.slots.shape[1]

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, self.epoch, i) if self.train else i)
        cand = []  # (slot, uid, n, centre)
        for s, uid in enumerate(self.slots.iloc[i].to_numpy()):
            if not isinstance(uid, str):
                continue
            n = self.n[uid]
            cs = np.arange(n) if self.train else np.unique(np.round(np.linspace(0, n - 1, min(n, self.k_eval))).astype(int))
            cand += [(s, uid, n, int(c)) for c in cs]
        if self.train and self.strat:  # equal share of the k windows per available slot, random centres
            by_slot = {}
            for c in cand:
                by_slot.setdefault(c[0], []).append(c)
            per = max(1, self.k // len(by_slot))
            cand = [c for g in by_slot.values() for c in (g[j] for j in sorted(rng.choice(len(g), min(per, len(g)), replace=False)))][: self.k]
        elif self.train:
            cand = [cand[j] for j in sorted(rng.choice(len(cand), min(self.k, len(cand)), replace=False))]
        x = np.zeros((self.K, 3, self.size, self.size), np.uint8)
        slot, pos, mask = np.zeros(self.K, np.int64), np.zeros(self.K, np.float32), np.zeros(self.K, bool)
        for uid in dict.fromkeys(c[1] for c in cand):
            path = str(self.dir / f"{uid}.b2nd")
            arr = blosc2.open(path)
            rows = [(j, c) for j, c in enumerate(cand) if c[1] == uid]
            n = rows[0][1][2]
            need = sorted({min(max(c[3] + d, 0), n - 1) for _, c in rows for d in (-1, 0, 1)})
            sl = dict(zip(need, [arr[k] for k in need]))
            for j, c in rows:
                x[j] = np.stack([sl[min(max(c[3] + d, 0), n - 1)] for d in (-1, 0, 1)])
                slot[j], pos[j], mask[j] = c[0], c[3] / max(n - 1, 1), True
            drop_cache(path)
        out = {"x": torch.from_numpy(x), "slot": torch.from_numpy(slot), "pos": torch.from_numpy(pos), "mask": torch.from_numpy(mask), "i": i}
        if self.Y is not None:
            out["y"], out["w"] = torch.tensor(self.Y[i]), torch.tensor(self.W[i])
        if self.aux is not None:
            out["aux"] = torch.tensor(self.aux[i])
        return out
