"""
Модуль загрузки данных СберИндекса.

Отвечает за:
- чтение parquet-файлов из data/raw/
- приведение к единому формату
- сохранение в data/interim/

Запуск:
    python -m src.data.load
"""

from pathlib import Path
import pandas as pd


# Пути к данным
RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")


def load_consumption() -> pd.DataFrame:
    """
    Загружает потребительские расходы по МО.

    Файл: data/raw/consumption_mo.parquet
    Столбцы: period, value, obs_status, source, category_15, mo, freq, ...
    Период: 2023-01 — 2024-12
    МО: 2118
    Категорий: 6
    """
    path = RAW_DIR / "consumption_mo.parquet"
    df = pd.read_parquet(path)

    # Приводим типы
    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Оставляем только нужные столбцы
    df = df[["period", "mo", "category_15", "value"]].copy()
    df = df.rename(columns={"category_15": "category"})

    return df


def load_mobility() -> pd.DataFrame:
    """
    Загружает индекс мобильности по МО.

    Файл: data/raw/mobility_mo.parquet
    Столбцы: period, value, obs_status, source, ref_area, ...
    Период: 2024-12 — 2025-11
    МО: 297
    """
    path = RAW_DIR / "mobility_mo.parquet"
    df = pd.read_parquet(path)

    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Переименовываем ref_area в mo
    df = df[["period", "ref_area", "value"]].copy()
    df = df.rename(columns={"ref_area": "mo"})

    return df


def main():
    """Точка входа: загружает всё и сохраняет в data/interim/."""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    print("Загрузка расходов...")
    consumption = load_consumption()
    print(f"  Строк: {len(consumption)}")
    print(f"  МО: {consumption['mo'].nunique()}")
    print(f"  Период: {consumption['period'].min()} — {consumption['period'].max()}")
    consumption.to_parquet(INTERIM_DIR / "consumption.parquet", index=False)

    print()
    print("Загрузка мобильности...")
    mobility = load_mobility()
    print(f"  Строк: {len(mobility)}")
    print(f"  МО: {mobility['mo'].nunique()}")
    print(f"  Период: {mobility['period'].min()} — {mobility['period'].max()}")
    mobility.to_parquet(INTERIM_DIR / "mobility.parquet", index=False)

    print()
    print("Готово. Файлы сохранены в data/interim/")


if __name__ == "__main__":
    main()
