'''
Data loading and feature engineering module for the classification task.
Functions are written to take/return dataframes so they can be unit tested
against small synthetic DataFrames.
'''

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F 

from src.schema import TRAIN_VALUES_SCHEMA, TRAIN_LABELS_SCHEMA

# Drop columns which add no predicitive signal (metadata)
DROP_COLS = [
    'id', 'wpt_name', 'subvillage', 'scheme_name', 'recorded_by', 'num_private', 'extraction_type_group', 'waterpoint_type_group', 'quantity_group', 'payment_type'
]

# High cardinality categorical columns
HIGH_CARDINALITY_COLS = ["funder", "installer", "ward", "lga"]
RARE_CATEGORY_THRESHOLD = 100  # min occurrences to keep as its own category


def load_raw(spark: SparkSession, values_path: str, labels_path: str) -> DataFrame:
    """Load the two raw CSVs with explicit schemas and join on id."""
    values = spark.read.option("header", True).schema(TRAIN_VALUES_SCHEMA).csv(values_path)
    labels = spark.read.option("header", True).schema(TRAIN_LABELS_SCHEMA).csv(labels_path)
    return values.join(labels, on="id", how="inner")


def bucket_rare_categories(df: DataFrame, col: str, threshold: int = RARE_CATEGORY_THRESHOLD) -> DataFrame:
    """Collapse categories occurring fewer than `threshold` times into 'OTHER'."""
    counts = df.groupBy(col).count()
    frequent_values = [row[col] for row in counts.filter(F.col("count") >= threshold).collect()]
    return df.withColumn(
        col,
        F.when(F.col(col).isin(frequent_values), F.col(col)).otherwise(F.lit("OTHER"))
    )


def engineer_features(df: DataFrame) -> DataFrame:
    """
    Apply cleaning and feature engineering:
      - parse date_recorded, derive year/month and pump age at recording time
      - treat 0/missing construction_year and gps_height as unknown (null) rather
        than literal 0, since 0 is a sentinel value in this dataset, not a real value
      - fill remaining nulls in categoricals with 'unknown' so indexers don't choke
      - drop redundant/leaky/free-text columns
    """
    df = df.withColumn("date_recorded", F.to_date("date_recorded"))
    df = df.withColumn("year_recorded", F.year("date_recorded"))
    df = df.withColumn("month_recorded", F.month("date_recorded"))

    df = df.withColumn(
        "construction_year",
        F.when(F.col("construction_year") == 0, None).otherwise(F.col("construction_year"))
    )
    df = df.withColumn(
        "pump_age_years",
        F.when(
            F.col("construction_year").isNotNull(),
            F.col("year_recorded") - F.col("construction_year")
        ).otherwise(None)
    )
    df = df.withColumn(
        "gps_height",
        F.when(F.col("gps_height") == 0, None).otherwise(F.col("gps_height"))
    )

    # Booleans with real nulls (not just False) carry a "not recorded" signal
    # in this dataset. Cast to string so "unknown" becomes its own category
    # rather than being silently merged into False by a boolean fillna.
    for bool_col in ["public_meeting", "permit"]:
        if bool_col in df.columns:
            df = df.withColumn(bool_col, F.col(bool_col).cast("string"))

    # construction_year is superseded by pump_age_years (which already
    # encodes the same information relative to date_recorded) and still
    # contains ~35% nulls (the 0-sentinel rows), so it is dropped rather
    # than imputed twice.
    df = df.drop("construction_year")

    string_cols = [f.name for f in df.schema.fields if str(f.dataType) == "StringType()"]
    df = df.fillna("unknown", subset=string_cols)

    numeric_fill_cols = ["pump_age_years", "gps_height", "population", "amount_tsh"]
    medians = df.approxQuantile(numeric_fill_cols, [0.5], 0.01)
    median_map = {c: m[0] for c, m in zip(numeric_fill_cols, medians)}
    df = df.fillna(median_map)

    existing_drop_cols = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(*existing_drop_cols)
    df = df.drop("date_recorded")  # superseded by year_recorded/month_recorded

    for col in HIGH_CARDINALITY_COLS:
        if col in df.columns:
            df = bucket_rare_categories(df, col)

    return df