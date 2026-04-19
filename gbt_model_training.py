import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import mlflow
import mlflow.sklearn

# Helper function to calculate and log all metrics
def evaluate_and_log(predictions, model_name):
    auc_roc = eval_roc.evaluate(predictions)
    auc_pr = eval_pr.evaluate(predictions)
    f1 = eval_multi.evaluate(predictions, {eval_multi.metricName: "f1"})
    precision = eval_multi.evaluate(predictions, {eval_multi.metricName: "weightedPrecision"})
    recall = eval_multi.evaluate(predictions, {eval_multi.metricName: "weightedRecall"})
    
    print(f"--- {model_name} Metrics on 2025 Test Data ---")
    print(f"AUC-ROC:   {auc_roc:.4f}")
    print(f"AUC-PR:    {auc_pr:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}\n")
    
    mlflow.log_metric("AUC_ROC", auc_roc)
    mlflow.log_metric("AUC_PR", auc_pr)
    mlflow.log_metric("F1_Score", f1)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)

# Remember to run this SQL statement before running the code below
# %sql
# CREATE VOLUME IF NOT EXISTS workspace.flights.mlflow_tmp;

# Define the volume path for MLflow temp storage
uc_volume_path = "/Volumes/workspace/flights/mlflow_tmp"

# 1. Feature Engineering Setup
categorical_cols = ["OP_UNIQUE_CARRIER_A", "ORIGIN_A", "DEST_A"]
numeric_cols = ["hub_congestion_hour", "travel_month", "scheduled_layover_mins", "is_same_airline"]

indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep") for c in categorical_cols]
encoders = [OneHotEncoder(inputCol=f"{c}_index", outputCol=f"{c}_vec") for c in categorical_cols]

# 2. VECTOR ASSEMBLER (Prep for Spark ML)
assembler_inputs = [f"{c}_vec" for c in categorical_cols] + numeric_cols
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# 3. Define the Evaluators (Capturing all the requested metrics)
eval_roc = BinaryClassificationEvaluator(labelCol="made_connection", metricName="areaUnderROC")
eval_pr = BinaryClassificationEvaluator(labelCol="made_connection", metricName="areaUnderPR")
eval_multi = MulticlassClassificationEvaluator(labelCol="made_connection", predictionCol="prediction")

print("Loading data from tables...")
# Load the pre-joined gold tables
train_final = spark.table("workspace.flights.ml_train_gold").sample(False, 0.1, seed=65)
test_final = spark.table("workspace.flights.ml_test_sampled").sample(False, 0.01, seed=65)

print("Converting Spark DataFrames to Pandas...")
# Convert to Pandas DataFrames
train_pd = train_final.toPandas()
test_pd = test_final.toPandas()

print(f"Train shape: {train_pd.shape}")
print(f"Test shape: {test_pd.shape}")

# Define features and target
categorical_features = ['OP_UNIQUE_CARRIER_A', 'ORIGIN_A', 'DEST_A']
numeric_features = ['hub_congestion_hour', 'travel_month', 'scheduled_layover_mins', 'is_same_airline']
feature_cols = categorical_features + numeric_features

X_train = train_pd[feature_cols]
y_train = train_pd['made_connection']
X_test = test_pd[feature_cols]
y_test = test_pd['made_connection']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', 'passthrough', numeric_features)
    ])

# Create full pipeline with GBT model
sklearn_pipeline = SklearnPipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(
        n_estimators=50,
        max_depth=5,
        learning_rate=0.1,
        random_state=67
    ))
])

print("\nTraining Scikit-Learn GBT model...")
with mlflow.start_run(run_name="GBT_SkLearn_NoSpark"):
    # Train the model
    sklearn_pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = sklearn_pipeline.predict(X_test)
    y_pred_proba = sklearn_pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"\n--- Scikit-Learn GBT Metrics ---")
    print(f"AUC-ROC:   {auc_roc:.4f}")
    print(f"AUC-PR:    {auc_pr:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    
    # Log metrics
    mlflow.log_metric("AUC_ROC", auc_roc)
    mlflow.log_metric("AUC_PR", auc_pr)
    mlflow.log_metric("F1_Score", f1)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    
    # Create signature for Unity Catalog
    from mlflow.models import infer_signature
    signature = infer_signature(X_train, y_pred_proba)
    
    # Log model (NO SPARK REQUIRED!)
    mlflow.sklearn.log_model(
        sk_model=sklearn_pipeline,
        artifact_path="flight_sklearn_model",
        signature=signature,
        input_example=X_train.head(5)
    )
    
    run_id = mlflow.active_run().info.run_id
    print(f"\n✅ Scikit-Learn model logged successfully!")
    print(f"Run ID: {run_id}")
