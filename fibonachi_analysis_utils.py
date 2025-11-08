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

def analyze_chain_fibonacci(chain_points: np.ndarray,
                            max_n: int = 10,
                            projection_mode: str = '2d') -> Dict[str, Any]:
    """
    Анализирует цепочку точек, используя динамическую классификацию и правила
    суммы Фибоначчи для дефляции, как запрошено пользователем.
    """
    PHI = (1 + 5 ** 0.5) / 2
    # Используем более длинный список меток для динамической классификации
    LABELS = ['L', 'M', 'S', 'XS', 'XXS', '3XS', '4XS', '5XS', '6XS', '7XS']
    max_n = len(LABELS)

    # 1. Рассчитываем длины сегментов (lengths)
    if projection_mode == 'x':
        lengths = np.abs(np.diff(chain_points[:, 1]))
    elif projection_mode == 'y':
        lengths = np.abs(np.diff(chain_points[:, 0]))
    else:
        lengths = np.linalg.norm(np.diff(chain_points, axis=0), axis=1)

    if lengths.size == 0:
        return {'segments': [], 'full_sequence_str': "", 'simplified_sequence_str': "", 'final_ls_ratio': np.nan,
                'fib_words': []}

    # 2. Находим базовую "L" (L_base)
    try:
        q90 = np.percentile(lengths, 90)
        valid_lengths = lengths[lengths >= q90]
        L_base = np.mean(valid_lengths) if valid_lengths.size > 0 else np.mean(lengths)
    except IndexError:
        L_base = np.mean(lengths)

    if not np.isfinite(L_base) or L_base <= 1e-6:
        L_base = np.max(lengths) if lengths.size > 0 else 1.0

    # 3. Генерируем прототипы и ДИНАМИЧЕСКИЕ допуски
    PROTOTYPES = [L_base * (PHI ** -n) for n in range(max_n)]
    TOLERANCES = []

    # Расчет допусков: 1.1 * половина расстояния до соседнего прототипа
    for n in range(max_n):
        if n == 0:
            if max_n > 1:
                tol_next = (PROTOTYPES[0] - PROTOTYPES[1]) / 2.0
            else:
                tol_next = (PROTOTYPES[0] - (PROTOTYPES[0] / PHI)) / 2.0
            TOLERANCES.append(tol_next * 1.1)
        elif n < max_n - 1:
            tol_hi = (PROTOTYPES[n - 1] - PROTOTYPES[n]) / 2.0
            tol_lo = (PROTOTYPES[n] - PROTOTYPES[n + 1]) / 2.0
            TOLERANCES.append(max(tol_hi, tol_lo) * 1.1)
        else:  # Последний элемент
            tol_hi = (PROTOTYPES[n - 1] - PROTOTYPES[n]) / 2.0
            tol_lo = (PROTOTYPES[n] - (PROTOTYPES[n] / PHI)) / 2.0
            TOLERANCES.append(max(tol_hi, tol_lo) * 1.1)

    if not TOLERANCES and PROTOTYPES:
        TOLERANCES.append((PROTOTYPES[0] - (PROTOTYPES[0] / PHI)) / 2.0 * 1.1)

    if len(TOLERANCES) < len(PROTOTYPES):
        missing = len(PROTOTYPES) - len(TOLERANCES)
        for i in range(missing):
            n_idx = len(TOLERANCES)
            tol_lo = (PROTOTYPES[n_idx] - (PROTOTYPES[n_idx] / PHI)) / 2.0
            TOLERANCES.append(tol_lo * 1.1)

    # 4. Классифицируем каждый сегмент
    full_sequence_data = []
    max_used_n = 0
    for length in lengths:
        dists = [abs(length - p) for p in PROTOTYPES]
        best_n = int(np.argmin(dists))

        if best_n < len(TOLERANCES) and dists[best_n] < TOLERANCES[best_n]:
            full_sequence_data.append({'len_1d': length, 'label': LABELS[best_n], 'n': best_n})
            max_used_n = max(max_used_n, best_n)
        else:
            full_sequence_data.append({'len_1d': length, 'label': '?', 'n': -1})

    full_sequence_labels = [s['label'] for s in full_sequence_data]

    # 5. ДИНАМИЧЕСКАЯ ДЕФЛЯЦИЯ по правилу (Smallest + Next Smallest) -> Next Larger
    # [ИСПРАВЛЕННАЯ ЛОГИКА - итеративная обработка каждого уровня]

    temp_labels = full_sequence_labels.copy()

    # n_pass от самого маленького используемого класса (S, XS, XXS) вверх к L (L=0)
    for n_level in range(max_used_n, 0, -1):

        # Индексы: S_sub (самый маленький, n_level) + M_sub (следующий, n_level - 1)
        # -> L_target (на уровень выше, n_level - 2)

        S_sub = LABELS[n_level]  # Сегмент S_sub (меньший из пары)
        M_sub = LABELS[n_level - 1]  # Сегмент M_sub (больший из пары)
        L_target = LABELS[n_level - 2] if n_level > 1 else LABELS[0]  # Сегмент L_target (результат суммы)

        # Фиксированная точка: повторять проход на текущем уровне n_level, пока есть изменения
        while True:
            i = 0
            pass_labels = []
            has_changed = False
            while i < len(temp_labels):

                # Проверяем, есть ли пара M_sub + S_sub (M_sub - Smallest + S_sub - Next Smallest)
                # Это фактически M_sub (L_n-1) + S_sub (L_n) -> L_target (L_n-2)
                # Пример: M (L1) + S (L2) -> L (L0)

                # Проверяем M_sub + S_sub
                if (i + 1 < len(temp_labels) and
                        temp_labels[i] == M_sub and temp_labels[i + 1] == S_sub):
                    pass_labels.append(L_target)
                    i += 2
                    has_changed = True

                # Проверяем S_sub + M_sub
                elif (i + 1 < len(temp_labels) and
                      temp_labels[i] == S_sub and temp_labels[i + 1] == M_sub):
                    pass_labels.append(L_target)
                    i += 2
                    has_changed = True

                else:
                    pass_labels.append(temp_labels[i])
                    i += 1

            # Если изменений нет, переходим к следующему уровню (break), иначе повторяем
            temp_labels = pass_labels
            if not has_changed:
                break

    simplified_sequence_labels = temp_labels

    # 6. Финальное L/S отображение (L и S)
    # Здесь мы оставляем только L и S, отбрасывая другие типы, если они остались.
    final_ls_labels = [label for label in simplified_sequence_labels if label in ['L', 'S']]
    l_count = final_ls_labels.count('L')
    s_count = final_ls_labels.count('S')

    final_ls_ratio = l_count / s_count if s_count > 0 else np.nan

    words = gen_fibonacci_words(len(lengths))

    return {
        'segments': full_sequence_data,
        'full_sequence_str': "-".join(full_sequence_labels),
        'simplified_sequence_str': "-".join(simplified_sequence_labels),
        'final_ls_ratio': final_ls_ratio,
        'fib_words': words
    }


def cluster_lengths(lengths: np.ndarray):
    # (Эта функция остается без изменений)
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
    # (Эта функция остается без изменений)
    if n <= 0: return []
    seq = [1, 1]
    while seq[-1] < n:
        seq.append(seq[-1] + seq[-2])
    return [k for k in seq if k <= n]


def gen_fibonacci_words(max_len: int, start: str = "L") -> List[str]:
    # (Эта функция остается без изменений)
    if max_len <= 0: return []
    words = ["L" if start.upper() == "L" else "S"]
    while len(words[-1]) <= max_len:
        nxt = "".join(("LS" if ch == "L" else "L") for ch in words[-1])
        if len(nxt) > max_len: break
        words.append(nxt)
    return words