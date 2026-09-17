"""
Модуль соединения основного датасета (features) с инфраструктурой.

Стратегия обработки пропусков (гибридная):
1. Логические нули — для газа и негазифицированных пунктов.
2. Медианная импутация по региону — для остальных признаков.
3. Флаги пропусков — где импутация не применялась.

Результат:
    data/processed/features_full.parquet

Запуск:
    python -m src.data.merge_features
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROCESSED_DIR = Path("data/processed")

# Признаки, где пропуск = отсутствие инфраструктуры (заполняем нулём)
ZERO_FILL = [
    "gas_network_m",
    "n_no_gas_settlements",
]

# Признаки, где импутируем медианой по региону
MEDIAN_IMPUTE_BY_REGION = [
    "water_network_m",
    "roads_km",
    "living_area",
    "invest_budget",
    "invest_org",
    "area_ha",
    "n_kindergarten",
    "n_families_subsidies",
    "subsidies_rub",
    "n_beneficiaries",
]

# Признаки, для которых создаём флаг "был пропуск"
FLAG_MISSING = [
    "n_families_subsidies",
    "subsidies_rub",
    "n_beneficiaries",
    "invest_budget",
]


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Создаёт флаги для признаков с потенциально значимыми пропусками."""
    df = df.copy()
    for col in FLAG_MISSING:
        if col in df.columns:
            df[f"flag_missing_{col}"] = df[col].isna().astype(int)
    return df


def fill_logical_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Заполняет нулями там, где пропуск = отсутствие инфраструктуры."""
    df = df.copy()
    for col in ZERO_FILL:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df


def impute_median_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Импутирует пропуски медианой по региону.
    Если в регионе все пропуски — использует глобальную медиану.
    """
    df = df.copy()
    for col in MEDIAN_IMPUTE_BY_REGION:
        if col not in df.columns:
            continue
        # Медиана по региону
        region_median = df.groupby("region_name")[col].transform("median")
        # Глобальная медиана
        global_median = df[col].median()
        # Сначала заполняем региональной, потом — глобальной
        df[col] = df[col].fillna(region_median).fillna(global_median)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет производные признаки."""
    df = df.copy()

    # Площадь в км² (данные в гектарах)
    df["area_km2"] = df["area_ha"] / 100

    # Плотность инфраструктуры (метры сети на км²)
    df["gas_density"] = df["gas_network_m"] / df["area_km2"]
    df["water_density"] = df["water_network_m"] / df["area_km2"]
    df["roads_density"] = df["roads_km"] / df["area_km2"]

    # Прокси населения через площадь жилья (~25 м²/чел, данные в тыс. м²)
    df["pop_est_living"] = (df["living_area"] * 1000) / 25

    # Душевые инвестиции (руб. на "прокси-жителя")
    df["invest_org_per_capita"] = df["invest_org"] / df["pop_est_living"]
    df["invest_budget_per_capita"] = df["invest_budget"] / df["pop_est_living"]

    return df


def main():
    print("Загрузка features...")
    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    print(f"  Строк: {len(features)}, столбцов: {features.shape[1]}")

    print()
    print("Загрузка infrastructure...")
    infra = pd.read_parquet(PROCESSED_DIR / "infrastructure.parquet")
    print(f"  Строк: {len(infra)}, столбцов: {infra.shape[1]}")

    print()
    print("Шаг 1. Флаги пропусков...")
    infra = add_missing_flags(infra)

    print("Шаг 2. Логические нули...")
    infra = fill_logical_zeros(infra)

    print("Шаг 3. Медианная импутация по региону...")
    infra = impute_median_by_region(infra)

    print("Шаг 4. Производные признаки...")
    infra = add_derived_features(infra)

    print()
    print("Соединение по oktmo_stable (inner join)...")
    merged = features.merge(
        infra.drop(columns=["region_name"]),
        on="oktmo_stable",
        how="inner",
    )
    print(f"  Строк: {len(merged)}, столбцов: {merged.shape[1]}")

    print()
    print("Пропуски в финальном датасете:")
    missing = merged.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) == 0:
        print("  Пропусков нет!")
    else:
        print(missing.to_string())

    print()
    print(f"Число МО: {merged['oktmo_stable'].nunique()}")
    if "is_metropolitan" in merged.columns:
        print(f"Мегаполисов (Мск + СПб): {merged['is_metropolitan'].sum()}")
    print()
    print("С флагами пропусков:")
    for col in merged.columns:
        if col.startswith("flag_missing_"):
            print(f"  {col}: {merged[col].sum()}")

    merged.to_parquet(PROCESSED_DIR / "features_full.parquet", index=False)
    print()
    print("Готово. Файл сохранён в data/processed/features_full.parquet")


if __name__ == "__main__":
    main()
