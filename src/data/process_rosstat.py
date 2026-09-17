"""
Модуль обработки данных Росстата.

Превращает «длинные» таблицы (строка на МО × год × период)
в «широкие» (одна строка на МО с усреднёнными значениями за 2023-2024).

Запуск:
    python -m src.data.process_rosstat
"""

from pathlib import Path

import pandas as pd


INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")


def clean_rosstat(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """
    Оставляет только:
    - МО верхнего уровня
    - период 'Январь-декабрь' (годовой итог)
    - годы 2023-2024
    И усредняет значения по годам.
    
    Возвращает таблицу: oktmo_stable | region_name | municipality | value_name
    """
    df = df[df["mun_level"] == "Муниципальное образование верхнего уровня"].copy()
    df = df[df["indicator_period"] == "Январь-декабрь"].copy()
    df = df[df["year"].isin([2023, 2024])].copy()

    # Усредняем по годам для каждого МО
    agg = (
        df.groupby(["oktmo_stable", "region_name", "municipality"], as_index=False)
        .agg(**{value_name: ("indicator_value", "mean")})
    )
    return agg


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Обработка зарплат...")
    wages = pd.read_parquet(INTERIM_DIR / "rosstat_wages.parquet")
    wages_clean = clean_rosstat(wages, "avg_wage")
    print(f"  Строк: {len(wages_clean)}")
    print(f"  МО: {wages_clean['oktmo_stable'].nunique()}")
    wages_clean.to_parquet(PROCESSED_DIR / "rosstat_wages_clean.parquet", index=False)

    print()
    print("Обработка численности...")
    employment = pd.read_parquet(INTERIM_DIR / "rosstat_employment.parquet")
    employment_clean = clean_rosstat(employment, "employment")
    print(f"  Строк: {len(employment_clean)}")
    print(f"  МО: {employment_clean['oktmo_stable'].nunique()}")
    employment_clean.to_parquet(PROCESSED_DIR / "rosstat_employment_clean.parquet", index=False)

    print()
    print("Готово. Файлы сохранены в data/processed/")


if __name__ == "__main__":
    main()
