from __future__ import annotations
import sys, json, math
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

# Обратите внимание: импорт preproc нужен для PreprocSettings
from preproc import PreprocSettings

# --- Функции поиска и загрузки данных ---

def _candidate_dirs(extra_image: Optional[Path]) -> List[Path]:
    cands: List[Path] = []
    try:
        cands.append(Path.cwd())
    except Exception:
        pass
    if getattr(sys, "frozen", False):
        try:
            cands.append(Path(sys.executable).resolve().parent)
        except Exception:
            pass
        try:
            cands.append(Path(getattr(sys, "_MEIPASS")))
        except Exception:
            pass
    else:
        try:
            cands.append(Path(__file__).resolve().parent)
        except Exception:
            pass
    if extra_image:
        try:
            cands.append(extra_image.resolve().parent)
        except Exception:
            pass
    uniq, seen = [], set()
    for d in cands:
        rp = str(d.resolve())
        if rp not in seen:
            uniq.append(d);
            seen.add(rp)
    return uniq

def _autofind_json(extra_image: Optional[Path]) -> Optional[Path]:
    pats = ["fibo_input.json", "*fibo*input*.json", "*.fibo.json", "*.json"]
    for base in _candidate_dirs(extra_image):
        for pat in pats:
            try:
                for p in base.glob(pat):
                    name = p.name.lower()
                    if "fibo" in name and "input" in name:
                        return p.resolve()
                    if pat == "*.json":
                        try:
                            obj = json.loads(p.read_text(encoding="utf-8"))
                            if isinstance(obj, dict) and "image" in obj and "points" in obj:
                                return p.resolve()
                        except Exception:
                            pass
            except Exception:
                continue
    return None

def find_default_json(base_dir: Path) -> Optional[Path]:
    cand = base_dir / "fibo_input.json"
    if cand.exists():
        try:
            d = json.loads(cand.read_text(encoding="utf-8"))
            if "image" in d and "points" in d:
                return cand.resolve()
        except Exception:
            pass
    for p in base_dir.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if "image" in d and "points" in d:
                return p.resolve()
        except Exception:
            pass
    return _autofind_json(None)

def load_input(json_path: Path):
    d = json.loads(json_path.read_text(encoding='utf-8'))
    img = Path(d['image']) if d.get('image') else None
    if not img:
        raise RuntimeError("JSON does not contain the key 'image'.")
    pts = np.array([[float(p['y']), float(p['x'])] for p in d.get('points', [])], float)
    center = None;
    dead = 0.0;
    srch = 0.0
    if isinstance(d.get('centers'), dict):
        c = d['centers'].get('overlay') or d['centers'].get('geometric')
        if c and 'x' in c and 'y' in c:
            center = (float(c['y']), float(c['x']))
    if isinstance(d.get('radii'), dict):
        if d['radii'].get('dead') is not None:  dead = float(d['radii']['dead'])
        if d['radii'].get('search') is not None: srch = float(d['radii']['search'])
    fallback_mode = d.get('preproc_mode')
    if not isinstance(fallback_mode, str):
        fallback_mode = None
    preproc = PreprocSettings.from_json(d.get('preproc'), fallback_mode=fallback_mode)
    return img, pts, center, dead, srch, preproc

# --- Функции алгоритмов анализа ---

def cluster_lengths(lengths: np.ndarray):
    if lengths.size == 0:
        return np.array([], dtype=int), float("nan"), float("nan"), 0, 1
    c0, c1 = float(lengths.min()), float(lengths.max())
    if c0 == c1:
        lab = np.zeros(len(lengths), dtype=int)
        return lab, c0, float("nan"), 0, 1
    lab = np.zeros(len(lengths), dtype=int)
    for _ in range(60):
        d0 = np.abs(lengths - c0)
        d1 = np.abs(lengths - c1)
        lab = (d1 < d0).astype(int)
        nc0 = float(lengths[lab == 0].mean()) if np.any(lab == 0) else c0
        nc1 = float(lengths[lab == 1].mean()) if np.any(lab == 1) else c1
        if abs(nc0 - c0) < 1e-6 and abs(nc1 - c1) < 1e-6:
            c0, c1 = nc0, nc1;
            break
        c0, c1 = nc0, nc1
    m0 = float(lengths[lab == 0].mean()) if np.any(lab == 0) else float("nan")
    m1 = float(lengths[lab == 1].mean()) if np.any(lab == 1) else float("nan")
    if (not math.isnan(m0)) and (not math.isnan(m1)) and m0 > m1:
        lab = 1 - lab
        m0, m1 = m1, m0
    return lab, m0, m1, 0, 1

def fib_list_upto(n: int) -> List[int]:
    if n <= 0: return []
    seq = [1, 1]
    while seq[-1] < n:
        seq.append(seq[-1] + seq[-2])
    return [k for k in seq if k <= n]

def gen_fibonacci_words(max_len: int, start: str = "L") -> List[str]:
    if max_len <= 0: return []
    words = ["L" if start.upper() == "L" else "S"]
    while len(words[-1]) <= max_len:
        nxt = "".join(("LS" if ch == "L" else "L") for ch in words[-1])
        if len(nxt) > max_len: break
        words.append(nxt)
    return words