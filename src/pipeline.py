"""
Builds the pyspark.ml Pipeline.
- Categorical columns are indexed then one-hot encoded rather than passed
  as raw indices.
- Numeric and one-hot-encoded features are assembled and scaled
  separately before a final assembly step, since scaling a mixed
  numeric/one-hot vector would incorrectly rescale the binary indicator
  columns.
- RandomForestClassifier is used because it handles multi-class targets
  natively, is robust to unscaled/skewed numeric inputs, and gives
  interpretable feature importances for the Results section - relevant
  given the class imbalance in this dataset.
- TrainValidationSplit (a single hold-out split) is used for
  hyperparameter tuning rather than full k-fold CrossValidator, trading
  a small amount of estimate stability for a large reduction in training
  time.
"""

from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

LABEL_COL = "status_group"


def get_feature_columns(df: DataFrame, label_col: str = LABEL_COL):
    """Split columns into categorical vs numeric feature lists based on dtype."""
    categorical_cols, numeric_cols = [], []
    for field in df.schema.fields:
        if field.name == label_col:
            continue
        dtype = str(field.dataType)
        if dtype == "StringType()":
            categorical_cols.append(field.name)
        else:
            numeric_cols.append(field.name)
    return categorical_cols, numeric_cols


def build_pipeline(categorical_cols, numeric_cols, label_col: str = LABEL_COL) -> Pipeline:
    label_indexer = StringIndexer(inputCol=label_col, outputCol="label", handleInvalid="keep")

    cat_idx_cols = [f"{c}_idx" for c in categorical_cols]
    cat_ohe_cols = [f"{c}_ohe" for c in categorical_cols]

    cat_indexers = [
        StringIndexer(inputCol=c, outputCol=idx, handleInvalid="keep")
        for c, idx in zip(categorical_cols, cat_idx_cols)
    ]
    encoder = OneHotEncoder(inputCols=cat_idx_cols, outputCols=cat_ohe_cols)

    numeric_assembler = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features")
    scaler = StandardScaler(
        inputCol="numeric_features", outputCol="numeric_features_scaled",
        withMean=True, withStd=True
    )

    final_assembler = VectorAssembler(
        inputCols=cat_ohe_cols + ["numeric_features_scaled"], outputCol="features"
    )

    classifier = RandomForestClassifier(
        featuresCol="features", labelCol="label", seed=42
    )

    stages = [label_indexer] + cat_indexers + [encoder, numeric_assembler, scaler, final_assembler, classifier]
    return Pipeline(stages=stages)


def build_tuned_estimator(pipeline: Pipeline):
    """Wrap the pipeline in TrainValidationSplit for hyperparameter tuning."""
    classifier = pipeline.getStages()[-1]

    # Grid kept deliberately small (2 configurations) to fit reasonable
    # training time on modest hardware; on a machine with more cores this
    # can be widened, e.g. numTrees: [50, 100, 200], maxDepth: [10, 15, 20].
    param_grid = (
        ParamGridBuilder()
        .addGrid(classifier.numTrees, [50, 100])
        .addGrid(classifier.maxDepth, [10])
        .build()
    )

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )

    return TrainValidationSplit(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        trainRatio=0.8,
        seed=42,
    )