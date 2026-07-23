"""
Tests for src.data_prep.

Each test builds a small, explicitly-schema'd in-memory DataFrame rather
than loading the full CSV, so the suite runs in seconds and each
test isolates one specific behaviour of engineer_features() /
bucket_rare_categories() rather than depending on the full dataset.
"""

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, BooleanType
)
from chispa.dataframe_comparer import assert_df_equality

from src.data_prep import engineer_features, bucket_rare_categories

# Minimal schema covering only the columns engineer_features() directly
# references (it also accepts and passes through other columns, but these
# are the ones with non-trivial transformation logic worth testing).
MINIMAL_SCHEMA = StructType([
    StructField("date_recorded", StringType(), True),
    StructField("construction_year", IntegerType(), True),
    StructField("gps_height", IntegerType(), True),
    StructField("public_meeting", BooleanType(), True),
    StructField("permit", BooleanType(), True),
    StructField("population", IntegerType(), True),
    StructField("amount_tsh", DoubleType(), True),
    StructField("funder", StringType(), True),
])


def make_df(spark, rows):
    return spark.createDataFrame(rows, schema=MINIMAL_SCHEMA)


def test_engineer_features_leaves_no_nulls(spark):
    rows = [
        ("2013-05-10", 1999, 1200, True, True, 100, 500.0, "Danida"),
        ("2015-01-01", 0, 0, None, None, 0, 0.0, "Danida"),  # sentinel/missing values
    ]
    df = make_df(spark, rows)
    result = engineer_features(df)

    null_counts = result.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in result.columns
    ]).collect()[0].asDict()

    assert all(v == 0 for v in null_counts.values()), f"Unexpected nulls: {null_counts}"


def test_zero_sentinel_construction_year_dropped_in_favour_of_pump_age(spark):
    rows = [("2015-01-01", 2000, 500, True, True, 50, 100.0, "Danida")]
    df = make_df(spark, rows)
    result = engineer_features(df)

    assert "construction_year" not in result.columns
    assert "pump_age_years" in result.columns


def test_pump_age_years_computed_correctly(spark):
    rows = [("2015-01-01", 2000, 500, True, True, 50, 100.0, "Danida")]
    df = make_df(spark, rows)
    result = engineer_features(df).collect()[0]

    assert result["pump_age_years"] == 15


def test_zero_sentinel_gps_height_treated_as_missing_and_imputed(spark):
    rows = [
        ("2015-01-01", 2000, 500, True, True, 50, 100.0, "Danida"),
        ("2015-01-01", 2000, 0, True, True, 50, 100.0, "Danida"),  # sentinel 0
    ]
    df = make_df(spark, rows)
    result = engineer_features(df).collect()
    heights = [r["gps_height"] for r in result]

    assert 0 not in heights, "sentinel 0 should have been imputed, not left as a literal height"
    assert all(h is not None for h in heights)


def test_boolean_nulls_become_unknown_string_category(spark):
    rows = [
        ("2015-01-01", 2000, 500, True, True, 50, 100.0, "Danida"),
        ("2015-01-01", 2000, 500, None, None, 50, 100.0, "Danida"),
    ]
    df = make_df(spark, rows)
    result = engineer_features(df)
    values = {r["public_meeting"] for r in result.collect()}

    assert "unknown" in values
    assert None not in values


def test_bucket_rare_categories_collapses_infrequent_values(spark):
    rows = [("A",), ("A",), ("A",), ("B",)]
    schema = StructType([StructField("funder", StringType(), True)])
    df = spark.createDataFrame(rows, schema=schema)

    result = bucket_rare_categories(df, "funder", threshold=2)

    expected = spark.createDataFrame([("A",), ("A",), ("A",), ("OTHER",)], schema=schema)
    assert_df_equality(result, expected, ignore_row_order=True)


def test_bucket_rare_categories_keeps_all_frequent_values(spark):
    rows = [("A",), ("A",), ("B",), ("B",)]
    schema = StructType([StructField("funder", StringType(), True)])
    df = spark.createDataFrame(rows, schema=schema)

    result = bucket_rare_categories(df, "funder", threshold=2)

    values = {r["funder"] for r in result.collect()}
    assert values == {"A", "B"}