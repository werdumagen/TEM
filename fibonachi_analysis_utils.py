from __future__ import annotations
import sys, json, math
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

# Обратите внимание: импорт preproc нужен для PreprocSettings
from preproc import PreprocSettings


# --- Функции поиска и загрузки данных (без изменений) ---

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

# --- ИЗМЕНЕНО: Добавлен `projection_mode` ---
def analyze_chain_fibonacci(chain_points: np.ndarray,
                            max_n: int = 6,
                            projection_mode: str = '2d') -> Dict[str, Any]:
    """
    Анализирует цепочку точек на иерархию Фибоначчи L, M, S, ...
    и упрощает ее до базовой L/S последовательности.
    Работает в 2D, 1D-X или 1D-Y режиме.
    """
    PHI = (1 + 5 ** 0.5) / 2
    LABELS = ['L', 'M', 'S', 'XS', 'XXS', '3XS', '4XS', '5XS']  # Метки для n=0 до n=7

    # 1. Рассчитываем длины сегментов (lengths) В ЗАВИСИМОСТИ ОТ РЕЖИМА
    if projection_mode == 'x':
        lengths = np.abs(np.diff(chain_points[:, 1]))  # Только X
        print(f"Analyzing 1D (X) distances for {len(lengths)} segments.")
    elif projection_mode == 'y':
        lengths = np.abs(np.diff(chain_points[:, 0]))  # Только Y
        print(f"Analyzing 1D (Y) distances for {len(lengths)} segments.")
    else:  # '2d'
        lengths = np.linalg.norm(np.diff(chain_points, axis=0), axis=1)  # 2D
        print(f"Analyzing 2D distances for {len(lengths)} segments.")

    if lengths.size == 0:
        return {'segments': [], 'full_sequence_str': "", 'simplified_sequence_str': "", 'final_ls_ratio': np.nan,
                'fib_words': []}

    # +++ ИСПРАВЛЕННАЯ ЛОГИКА L_base +++
    # 2. Находим базовую "L" (n=0)
    try:
        # Используем 90-й процентиль для более надежного L_base
        q90 = np.percentile(lengths, 90)
        # Проверяем, есть ли вообще значения >= q90
        valid_lengths = lengths[lengths >= q90]
        if valid_lengths.size > 0:
            L_base = np.mean(valid_lengths)
        else:
            L_base = np.mean(lengths)  # Откат, если q90 не сработал
    except IndexError:
        L_base = np.mean(lengths)  # Откат, если lengths пустой

    if not np.isfinite(L_base) or L_base <= 1e-6:
        L_base = np.max(lengths) if lengths.size > 0 else 1.0
    # +++ КОНЕЦ ИСПРАВЛЕНИЯ +++

    # 3. Генерируем прототипы и допуски
    PROTOTYPES = [L_base * (PHI ** -n) for n in range(max_n)]

    # +++ ИСПРАВЛЕННАЯ ЛОГИКА ДОПУСКОВ +++
    TOLERANCES = []

    # Допуск для L (n=0)
    if max_n > 0:
        # Расстояние до M (n=1) или до L/PHI
        tol_0 = (PROTOTYPES[0] - PROTOTYPES[1]) / 2.0 if max_n > 1 else (PROTOTYPES[0] - (PROTOTYPES[0] / PHI)) / 2.0
        TOLERANCES.append(tol_0 * 1.1)  # Допуск = половина расстояния до M, +10%

    # Допуски для M, S, ... (n=1 до max_n-2)
    for n in range(1, max_n - 1):
        tol_hi = (PROTOTYPES[n - 1] - PROTOTYPES[n]) / 2.0  # Половина расстояния до "старшего"
        tol_lo = (PROTOTYPES[n] - PROTOTYPES[n + 1]) / 2.0  # Половина расстояния до "младшего"
        # Допуск = наибольшая из двух половин, +10% (чтобы окна перекрывались)
        TOLERANCES.append(max(tol_hi, tol_lo) * 1.1)

    # Допуск для последнего элемента (n=max_n-1)
    if max_n > 1:
        tol_hi = (PROTOTYPES[max_n - 2] - PROTOTYPES[max_n - 1]) / 2.0
        tol_lo = (PROTOTYPES[max_n - 1] - (PROTOTYPES[max_n - 1] / PHI)) / 2.0
        TOLERANCES.append(max(tol_hi, tol_lo) * 1.1)

    # Если max_n = 1, а TOLERANCES пуст
    if not TOLERANCES and PROTOTYPES:
        TOLERANCES.append((PROTOTYPES[0] - (PROTOTYPES[0] / PHI)) / 2.0 * 1.1)

    # Убедимся, что у нас есть допуск для каждого прототипа
    if len(TOLERANCES) < len(PROTOTYPES):
        missing = len(PROTOTYPES) - len(TOLERANCES)
        for i in range(missing):
            # Добавляем допуск для оставшихся
            n_idx = len(TOLERANCES)
            tol_lo = (PROTOTYPES[n_idx] - (PROTOTYPES[n_idx] / PHI)) / 2.0
            TOLERANCES.append(tol_lo * 1.1)
    # +++ КОНЕЦ ИСПРАВЛЕННОЙ ЛОГИКИ +++

    # 4. Классифицируем каждый сегмент
    full_sequence_data = []
    for length in lengths:
        dists = [abs(length - p) for p in PROTOTYPES]
        best_n = int(np.argmin(dists))

        # +++ ИСПРАВЛЕНИЕ: Проверяем, что best_n в пределах TOLERANCES +++
        if best_n < len(TOLERANCES) and dists[best_n] < TOLERANCES[best_n]:
            full_sequence_data.append({'len_1d': length, 'label': LABELS[best_n], 'n': best_n})
        else:
            full_sequence_data.append({'len_1d': length, 'label': '?', 'n': -1})

    full_sequence_labels = [s['label'] for s in full_sequence_data]

    # 5. Упрощаем ("дефляция")
    temp_labels = full_sequence_labels.copy()

    for n_pass in range(max_n - 3, -1, -1):  # т.е. от n=3 до n=0
        i = 0
        pass_labels = []
        L_n, M_n, S_n = LABELS[n_pass], LABELS[n_pass + 1], LABELS[n_pass + 2]  # n=0,1,2 -> L,M,S

        while i < len(temp_labels):
            if (i + 1 < len(temp_labels) and
                    temp_labels[i] == M_n and temp_labels[i + 1] == S_n):
                pass_labels.append(L_n)
                i += 2
            elif (i + 1 < len(temp_labels) and
                  temp_labels[i] == S_n and temp_labels[i + 1] == M_n):