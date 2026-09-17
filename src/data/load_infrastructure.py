"""
Модуль загрузки данных инфраструктуры из БД ПМО (разделы 6, 8, 9, 12, 34).

Читает parquet-файлы из data/raw/infrastructure/, фильтрует:
- по годам (2023-2024, совпадают с периодом СберИндекса)
- по нужным показателям

Группирует по oktmo_stable (усредняет по доступным годам).

Результат:
    data/processed/infrastructure.parquet

Запуск:
    python -m src.data.load_infrastructure
"""

from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw/infrastructure")
PROCESSED_DIR = Path("data/processed")

# Годы, совпадающие с периодом СберИндекса (2023-01 — 2024-12)
YEARS_KEEP = [2023, 2024]

# Показатели, которые берём (indicator_name -> новое имя столбца)
INDICATORS = {
    # Раздел 8. Коммунальная сфера
    "Общая площадь жилых помещений": "living_area",
    "Число проживающих в ветхих жилых домах": "n_dilapidated_residents",
    "Одиночное протяжение уличной газовой сети": "gas_network_m",
    "Одиночное протяжение уличной водопроводной сети": "water_network_m",
    "Количество негазифицированных населенных пунктов": "n_no_gas_settlements",

    # Раздел 6. Территория
    "Общая площадь земель муниципального образования": "area_ha",
    "Протяженность автодорог общего пользования местного значения": "roads_km",

    # Раздел 12. Социальная поддержка ЖКХ
    "Число семей, получавших субсидии на оплату жилого помещения и коммунальных услуг": "n_families_subsidies",
    "Сумма начисленных субсидий населению на оплату жилого помещения и коммунальных услуг": "subsidies_rub",
    "Численность граждан, пользующихся социальной поддержкой (льготами) по оплате жилого помещения и коммунальных услуг": "n_beneficiaries",

    # Раздел 34. Образование
    "Численность воспитанников, посещающих организации, осуществляющие образовательную деятельность по образовательным программам дошкольного образования, присмотр и уход за детьми": "n_kindergarten",
    "Численность обучающихся общеобразовательных организаций с учетом обособленных подразделений (филиалов)": "n_school_students",

    # Раздел 9. Инвестиции
    "Инвестиции в основной капитал, осуществляемые организациями, находящимися на территории муниципального образования (без субъектов малого предпринимательства)": "invest_org",
    "Инвестиции в основной капитал за счет средств местных бюджетов": "invest_budget",
}

# Файлы разделов
SECTION_FILES = {
    6: "data_section6_112_v20250918.parquet",
    8: "data_section8_112_v20250918.parquet",
    9: "data_section9_112_v20250918.parquet",
    12: "data_section12_112_v20250918.parquet",
    34: "data_section34_112_v20250918.parquet",
}


def load_section(section_num: int) -> pd.DataFrame:
    """Загружает один раздел и фильтрует по нужным показателям и годам."""
    path = RAW_DIR / SECTION_FILES[section_num]
    df = pd.read_parquet(path)

    df = df[df["year"].isin(YEARS_KEEP)].copy()
    df = df[df["mun_level"] == "Муниципальное образование верхнего уровня"].copy()
    df = df[df["indicator_name"].isin(INDICATORS.keys())].copy()

    return df


def build_infrastructure() -> pd.DataFrame:
    """Собирает все разделы в один датасет."""
    frames = []
    for sec in SECTION_FILES.keys():
        print(f"  Раздел {sec}...")
        df = load_section(sec)
        if not df.empty:
            df = df[["oktmo_stable", "region_name", "indicator_name", "year", "indicator_value"]]
            frames.append(df)
            print(f"    Строк: {len(df)}, показателей: {df['indicator_name'].nunique()}")

    all_df = pd.concat(frames, ignore_index=True)
    all_df["feature"] = all_df["indicator_name"].map(INDICATORS)

    # Усредняем по доступным годам
    grouped = all_df.groupby(
        ["oktmo_stable", "region_name", "feature"], as_index=False
    )["indicator_value"].mean()

    wide = grouped.pivot(
        index=["oktmo_stable", "region_name"], columns="feature", values="indicator_value"
    ).reset_index()

    return wide


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Загрузка инфраструктуры из разделов 6, 8, 9, 12, 34...")
    print()

    result = build_infrastructure()

    print()
    print(f"Финальный размер: {result.shape}")
    print(f"Столбцы: {result.columns.tolist()}")
    print()
    print(f"Уникальных МО: {result['oktmo_stable'].nunique()}")
    print()
    print("Пропуски:")
    print(result.isnull().sum())

    result.to_parquet(PROCESSED_DIR / "infrastructure.parquet", index=False)
    print()
    print("Готово. Файл сохранён в data/processed/infrastructure.parquet")


if __name__ == "__main__":
    main()
