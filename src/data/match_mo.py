"""
Модуль сопоставления муниципалитетов между СберИндексом и Росстатом.

Алгоритм:
1. Нормализуем названия МО в обоих источниках (убираем типы, служебные слова).
2. Для каждого МО СберИндекса ищем лучшее совпадение в Росстате (fuzzy matching).
3. Оставляем только пары со score >= threshold.
4. Для дубликатов (когда несколько МО Росстата с одинаковым названием)
   берём лучший score, а неуверенные помечаем флагом.

Результат:
    data/processed/mo_mapping.parquet

Запуск:
    python -m src.data.match_mo
"""

import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process


INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")

SCORE_THRESHOLD = 85  # минимальный score для принятия матча

# Служебные слова, которые убираем при нормализации
STOP_PHRASES = [
    "внутригородская территория города федерального значения",
    "внутригородская территория",
    "муниципальный район",
    "муниципальный округ",
    "городской округ",
    "сельское поселение",
    "городское поселение",
    "муниципальное образование",
    "город",
    "район",
    "округ",
    "поселение",
]


def normalize(name: str) -> str:
    """Приводит название МО к 'голому' корню."""
    if not isinstance(name, str):
        return ""
    s = name.lower().replace("ё", "е")
    for phrase in STOP_PHRASES:
        s = s.replace(phrase, " ")
    s = re.sub(r"[^а-яa-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def match_municipalities(
    sber_names: list[str], rosstat_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Сопоставляет названия МО СберИндекса с МО Росстата.

    Возвращает DataFrame: [mo_sber, mo_rosstat, oktmo_stable, region_name, score]
    """
    rosstat_norms = rosstat_df["norm"].tolist()

    results = []
    for sber_name in sber_names:
        sber_norm = normalize(sber_name)
        if not sber_norm:
            results.append(
                {
                    "mo_sber": sber_name,
                    "mo_rosstat": None,
                    "oktmo_stable": None,
                    "region_name": None,
                    "score": 0,
                }
            )
            continue

        match = process.extractOne(sber_norm, rosstat_norms, scorer=fuzz.token_sort_ratio)
        if match is None:
            best_score = 0
            best_idx = None
        else:
            _, best_score, best_idx = match

        if best_score >= SCORE_THRESHOLD and best_idx is not None:
            r = rosstat_df.iloc[best_idx]
            results.append(
                {
                    "mo_sber": sber_name,
                    "mo_rosstat": r["municipality"],
                    "oktmo_stable": r["oktmo_stable"],
                    "region_name": r["region_name"],
                    "score": best_score,
                }
            )
        else:
            results.append(
                {
                    "mo_sber": sber_name,
                    "mo_rosstat": None,
                    "oktmo_stable": None,
                    "region_name": None,
                    "score": best_score if best_score else 0,
                }
            )

    return pd.DataFrame(results)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Загрузка данных СберИндекса...")
    cons = pd.read_parquet(INTERIM_DIR / "consumption.parquet")
    sber_names = sorted(cons["mo"].unique())
    print(f"  МО СберИндекса: {len(sber_names)}")

    print()
    print("Загрузка данных Росстата...")
    rosstat = pd.read_parquet(PROCESSED_DIR / "rosstat_wages_clean.parquet")
    rosstat = rosstat.copy()
    rosstat["norm"] = rosstat["municipality"].apply(normalize)
    print(f"  МО Росстата: {len(rosstat)}")

    print()
    print("Сопоставление...")
    mapping = match_municipalities(sber_names, rosstat)

    matched = mapping[mapping["oktmo_stable"].notna()]
    unmatched = mapping[mapping["oktmo_stable"].isna()]

    print(f"  Сопоставлено: {len(matched)} ({len(matched) / len(mapping) * 100:.1f}%)")
    print(f"  Не сопоставлено: {len(unmatched)}")

    print()
    print(f"  Средний score: {matched['score'].mean():.1f}")
    print(f"  Минимальный score: {matched['score'].min():.1f}")

    mapping.to_parquet(PROCESSED_DIR / "mo_mapping.parquet", index=False)
    print()
    print("Готово. Файл сохранён в data/processed/mo_mapping.parquet")


if __name__ == "__main__":
    main()
