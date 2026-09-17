"""
Модуль сборки финального датасета для кластеризации.

Соединяет три источника по oktmo_stable:
- Расходы СберИндекса (5 категорий × 24 месяца → средние по МО)
- Зарплата Росстата (одно значение на МО)
- Численность Росстата (одно значение на МО)

Результат:
    data/processed/features.parquet

Запуск:
    python -m src.data.build_features
"""

from pathlib import Path

import pandas as pd


INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")


def build_consumption_features() -> pd.DataFrame:
    """
    Превращает 'длинную' таблицу расходов (МО × месяц × категория)
    в 'широкую' — одна строка на МО, столбцы = категории.
    
    Расходы усредняются по 24 месяцам.
    """
    cons = pd.read_parquet(INTERIM_DIR / "consumption.parquet")

    # Считаем долю каждой категории в среднем за период
    # Сначала средние расходы по категориям на МО
    avg = cons.groupby(["mo", "category"], as_index=False)["value"].mean()

    # Разворачиваем в широкий формат: строки = МО, столбцы = категории
    wide = avg.pivot(index="mo", columns="category", values="value").reset_index()

    # Переименуем столбцы для удобства
    wide = wide.rename(
        columns={
            "Продовольствие": "spend_food",
            "Общественное питание": "spend_catering",
            "Здоровье": "spend_health",
            "Транспорт": "spend_transport",
            "Маркетплейсы": "spend_marketplaces",
            "Все категории": "spend_total",
        }
    )

    # Считаем доли (структуру трат) — это важнее абсолютных значений
    spend_cols = [c for c in wide.columns if c.startswith("spend_") and c != "spend_total"]
    for col in spend_cols:
        share_col = col.replace("spend_", "share_")
        wide[share_col] = wide[col] / wide["spend_total"]

    # Добавляем логарифм общей суммы (для нормализации масштаба)
    wide["log_total"] = np.log1p(wide["spend_total"])

    return wide


def main():
    import numpy as np  # локальный импорт, чтобы не забыть

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Сборка признаков расходов...")
    cons_features = build_consumption_features()
    print(f"  МО: {len(cons_features)}")
    print(f"  Столбцы: {cons_features.columns.tolist()}")

    print()
    print("Загрузка маппинга...")
    mapping = pd.read_parquet(PROCESSED_DIR / "mo_mapping.parquet")
    mapping = mapping[mapping["oktmo_stable"].notna()].copy()
    print(f"  Сопоставленных МО: {len(mapping)}")

    print()
    print("Соединение расходов с маппингом...")
    # Соединяем по названию МО СберИндекса
    df = mapping.merge(
        cons_features,
        left_on="mo_sber",
        right_on="mo",
        how="inner",
    )
    print(f"  Строк: {len(df)}")

    print()
    print("Загрузка зарплат...")
    wages = pd.read_parquet(PROCESSED_DIR / "rosstat_wages_clean.parquet")
    wages = wages[["oktmo_stable", "avg_wage"]]
    df = df.merge(wages, on="oktmo_stable", how="left")
    print(f"  Строк с зарплатой: {df['avg_wage'].notna().sum()}")

    print()
    print("Загрузка численности...")
    emp = pd.read_parquet(PROCESSED_DIR / "rosstat_employment_clean.parquet")
    emp = emp[["oktmo_stable", "employment"]]
    df = df.merge(emp, on="oktmo_stable", how="left")
    print(f"  Строк с численностью: {df['employment'].notna().sum()}")

    # Убираем служебный столбец mo
    df = df.drop(columns=["mo"])

    print()
    print(f"Финальный размер: {df.shape}")
    print(f"Столбцы: {df.columns.tolist()}")

    df.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
    print()
    print("Готово. Файл сохранён в data/processed/features.parquet")


if __name__ == "__main__":
    import numpy as np
    main()
