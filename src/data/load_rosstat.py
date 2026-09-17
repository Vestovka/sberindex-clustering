"""
Модуль загрузки данных Росстата из архива БД ПМО.

Читает большие CSV прямо из ZIP-архива (потоково), фильтрует по годам,
сохраняет результат в data/interim/ в формате parquet.

Запуск:
    python -m src.data.load_rosstat
"""

import io
import zipfile
from pathlib import Path

import pandas as pd


# Пути
ZIP_PATH = Path.home() / "Downloads" / "data_section32_112_v20250918.zip"
INTERIM_DIR = Path("data/interim")

# Файлы внутри архива
FILES = {
    "employment": "data_Y48423005_112_v20250918.csv",  # численность работников
    "wages": "data_Y48423007_112_v20250918.csv",       # средняя зарплата
}

# Какие годы оставляем (совпадают с периодом СберИндекса 2023-2024)
YEARS_KEEP = {"2023", "2024"}

# Какие столбцы нужны
COLUMNS_KEEP = [
    "oktmo_stable",
    "region_name",
    "municipality",
    "mun_level",
    "okved",
    "year",
    "indicator_value",
    "indicator_unit",
    "indicator_period",
]


def read_filtered_csv(zip_path: Path, inner_name: str) -> pd.DataFrame:
    """
    Читает CSV из ZIP потоково, оставляет только строки за нужные годы
    и только нужные столбцы.

    Возвращает DataFrame.
    """
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        with z.open(inner_name) as f:
            reader = io.TextIOWrapper(f, encoding="utf-8")
            header = reader.readline().strip().split(";")
            idx = {col: header.index(col) for col in COLUMNS_KEEP if col in header}
            idx_year = header.index("year")

            for i, line in enumerate(reader):
                parts = line.rstrip("\n").split(";")
                if len(parts) <= idx_year:
                    continue
                if parts[idx_year] not in YEARS_KEEP:
                    continue
                # берём только нужные столбцы
                row = {col: parts[j] for col, j in idx.items()}
                rows.append(row)

    df = pd.DataFrame(rows)
    # Приводим типы
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["indicator_value"] = pd.to_numeric(df["indicator_value"], errors="coerce")
    return df


def load_employment() -> pd.DataFrame:
    """Загружает численность работников."""
    print("Загрузка численности работников (Y48123005)...")
    df = read_filtered_csv(ZIP_PATH, FILES["employment"])
    return df


def load_wages() -> pd.DataFrame:
    """Загружает среднемесячную зарплату."""
    print("Загрузка зарплат (Y48123007)...")
    df = read_filtered_csv(ZIP_PATH, FILES["wages"])
    return df


def main():
    """Точка входа: загружает оба показателя и сохраняет в data/interim/."""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    employment = load_employment()
    print(f"  Строк: {len(employment)}")
    print(f"  МО: {employment['oktmo_stable'].nunique()}")
    print(f"  Годы: {sorted(employment['year'].dropna().unique())}")
    employment.to_parquet(INTERIM_DIR / "rosstat_employment.parquet", index=False)

    print()

    wages = load_wages()
    print(f"  Строк: {len(wages)}")
    print(f"  МО: {wages['oktmo_stable'].nunique()}")
    print(f"  Годы: {sorted(wages['year'].dropna().unique())}")
    wages.to_parquet(INTERIM_DIR / "rosstat_wages.parquet", index=False)

    print()
    print("Готово. Файлы сохранены в data/interim/")


if __name__ == "__main__":
    main()
