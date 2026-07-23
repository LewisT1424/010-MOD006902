"""
Tests for src.pipeline. Uses a synthetic DataFrame shaped like the
engineered dataset (a mix of categorical and numeric columns plus a
status_group label) to confirm the pipeline builds and fits correctly,
without needing the full dataset or a lengthy training run.
"""

from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

from src.pipeline import get_feature_columns, build_pipeline

SCHEMA = StructType([
    StructField("region", StringType(), True),
    StructField("extraction_type", StringType(), True),
    StructField("amount_tsh", DoubleType(), True),
    StructField("population", IntegerType(), True),
    StructField("status_group", StringType(), True),
])

ROWS = [
    ("Iringa", "gravity", 500.0, 100, "functional"),
    ("Iringa", "handpump", 0.0, 50, "non functional"),
    ("Mbeya", "gravity", 250.0, 200, "functional"),
    ("Mbeya", "submersible", 1000.0, 10, "functional needs repair"),
    ("Dodoma", "handpump", 0.0, 5, "non functional"),
    ("Dodoma", "gravity", 300.0, 80, "functional"),
]


def make_df(spark):
    return spark.createDataFrame(ROWS, schema=SCHEMA)


def test_get_feature_columns_splits_by_dtype(spark):
    df = make_df(spark)
    cat_cols, num_cols = get_feature_columns(df)

    assert set(cat_cols) == {"region", "extraction_type"}
    assert set(num_cols) == {"amount_tsh", "population"}
    assert "status_group" not in cat_cols
    assert "status_group" not in num_cols


def test_pipeline_fits_and_predicts_without_error(spark):
    df = make_df(spark)
    cat_cols, num_cols = get_feature_columns(df)
    pipeline = build_pipeline(cat_cols, num_cols)

    model = pipeline.fit(df)
    predictions = model.transform(df)

    assert "prediction" in predictions.columns
    assert "label" in predictions.columns
    assert predictions.count() == df.count()


def test_pipeline_predictions_are_valid_class_indices(spark):
    df = make_df(spark)
    cat_cols, num_cols = get_feature_columns(df)
    pipeline = build_pipeline(cat_cols, num_cols)

    model = pipeline.fit(df)
    predictions = model.transform(df).collect()
    n_classes = df.select("status_group").distinct().count()

    for row in predictions:
        assert 0 <= row["prediction"] < n_classes