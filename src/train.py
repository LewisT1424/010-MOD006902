"""
End-to-end training script for the water pump status classifier.

Usage:
    python -m src.train

Produces (in reports/):
    metrics.json              - accuracy, F1, precision, recall
    confusion_matrix.json/png
    feature_importance.json/png
    class_distribution.json/png
    rdd_vs_dataframe.json     - timing comparison
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from src.data_prep import load_raw, engineer_features
from src.pipeline import get_feature_columns, build_pipeline, build_tuned_estimator
from src.rdd_demo import run_comparison
from src.spark_utils import get_spark

DATA_DIR = "data"
REPORTS_DIR = "reports"


def save_charts(class_dist, cm_rows, feature_importances):
    labels = ["functional", "non functional", "functional needs repair"]

    # class distribution
    class_dist_sorted = sorted(class_dist, key=lambda x: -x["count"])
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([c["status_group"] for c in class_dist_sorted],
                  [c["count"] for c in class_dist_sorted],
                  color=["#2b6cb0", "#e53e3e", "#dd6b20"])
    ax.set_ylabel("Number of water points")
    ax.set_title(f"Class distribution: water pump status (n={sum(c['count'] for c in class_dist):,})")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 500, f"{int(b.get_height()):,}", ha="center")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/class_distribution.png", dpi=150)
    plt.close()

    # confusion matrix
    mat = np.zeros((3, 3))
    for r in cm_rows:
        mat[labels.index(r["actual"]), labels.index(r["predicted"])] = r["count"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mat, cmap="Blues")
    short = ["functional", "non functional", "needs repair"]
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(short, rotation=30, ha="right"); ax.set_yticklabels(short)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (test set, n={int(mat.sum()):,})")
    for i in range(3):
        for j in range(3):
            pct = mat[i, j] / mat[i].sum() * 100
            color = "white" if mat[i, j] > mat.max() * 0.5 else "black"
            ax.text(j, i, f"{int(mat[i,j]):,}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=9)
    plt.colorbar(im, label="Count")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    # feature importance
    top = feature_importances[:12][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh([f["feature"] for f in top], [f["importance"] for f in top], color="#2b6cb0")
    ax.set_xlabel("Gini importance")
    ax.set_title("Top 12 feature importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/feature_importance.png", dpi=150)
    plt.close()


def extract_feature_importances(model: PipelineModel, transformed_sample, numeric_cols):
    attrs = transformed_sample.schema["features"].metadata["ml_attr"]["attrs"]
    all_attrs = []
    for group in attrs.values():
        all_attrs.extend(group)
    all_attrs.sort(key=lambda a: a["idx"])

    clean_names = []
    for a in all_attrs:
        name = a["name"]
        if name.startswith("numeric_features_scaled_"):
            idx = int(name.split("_")[-1])
            clean_names.append(numeric_cols[idx])
        else:
            clean_names.append(name.replace("_ohe_", ": "))

    rf_stage = model.stages[-1]
    importances = rf_stage.featureImportances.toArray()
    pairs = sorted(zip(clean_names, importances), key=lambda x: -x[1])
    return [{"feature": n, "importance": float(i)} for n, i in pairs]


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    spark = get_spark(app_name="waterpump-training")
    spark.sparkContext.setLogLevel("ERROR")

    print("Loading and engineering data...")
    df = load_raw(spark, f"{DATA_DIR}/Training_set_values.csv", f"{DATA_DIR}/Training_set_labels.csv")
    df = engineer_features(df)
    df.cache()
    print(f"Rows: {df.count():,}  Columns: {len(df.columns)}")

    class_dist = [{"status_group": r["status_group"], "count": r["count"]}
                  for r in df.groupBy("status_group").count().collect()]

    print("\nRunning RDD vs DataFrame comparison...")
    rdd_comparison = run_comparison(df)
    print(rdd_comparison)
    with open(f"{REPORTS_DIR}/rdd_vs_dataframe.json", "w") as f:
        json.dump(rdd_comparison, f, indent=2)

    cat_cols, num_cols = get_feature_columns(df)
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    train.cache(); test.cache()
    print(f"\nTrain rows: {train.count():,}  Test rows: {test.count():,}")

    print("\nBuilding and tuning pipeline (this can take a while on limited cores)...")
    pipeline = build_pipeline(cat_cols, num_cols)
    tvs = build_tuned_estimator(pipeline)

    t0 = time.time()
    tvs_model = tvs.fit(train)
    fit_time = time.time() - t0
    best_model = tvs_model.bestModel
    rf_stage = best_model.stages[-1]
    print(f"Tuning complete in {fit_time:.1f}s. Best: numTrees={rf_stage.getNumTrees}, maxDepth={rf_stage.getOrDefault('maxDepth')}")

    preds = best_model.transform(test)
    preds.cache()

    evaluators = {
        "accuracy": MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy"),
        "f1": MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1"),
        "weighted_precision": MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision"),
        "weighted_recall": MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall"),
    }
    metrics = {name: ev.evaluate(preds) for name, ev in evaluators.items()}
    metrics["tuned_fit_time_seconds"] = fit_time
    metrics["best_num_trees"] = rf_stage.getNumTrees
    metrics["best_max_depth"] = rf_stage.getOrDefault("maxDepth")
    print("\nMetrics:", metrics)
    with open(f"{REPORTS_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    label_map = {0.0: "functional", 1.0: "non functional", 2.0: "functional needs repair"}
    cm_rows_raw = preds.groupBy("status_group", "prediction").count().collect()
    cm_rows = [{"actual": r["status_group"], "predicted": label_map[r["prediction"]], "count": r["count"]}
               for r in cm_rows_raw]
    with open(f"{REPORTS_DIR}/confusion_matrix.json", "w") as f:
        json.dump(cm_rows, f, indent=2)

    print("\nExtracting feature importances...")
    sample_transformed = best_model.transform(test.limit(10))
    feature_importances = extract_feature_importances(best_model, sample_transformed, num_cols)
    with open(f"{REPORTS_DIR}/feature_importance.json", "w") as f:
        json.dump(feature_importances, f, indent=2)

    print("Saving charts...")
    save_charts(class_dist, cm_rows, feature_importances)

    best_model.write().overwrite().save(f"{REPORTS_DIR}/best_model")

    print("\nDone. Results written to reports/")
    spark.stop()


if __name__ == "__main__":
    main()