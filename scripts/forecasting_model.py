from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import col
from pyspark.ml.evaluation import RegressionEvaluator

def create_spark_session():
    spark = SparkSession.builder \
        .appName("apple-forecasting-model") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

def train_forecast_model(spark):
    """Train demand forecasting model"""
    
    print("\n" + "="*60)
    print("TRAINING DEMAND FORECASTING MODEL")
    print("="*60)
    
    # Read features
    df = spark.read.parquet("output/forecast_features")
    
    print(f"\nTraining set size: {df.count()} records")
    df.show(10)
    
    # Prepare features for ML
    assembler = VectorAssembler(
        inputCols=["prev_units", "growth_rate", "sentiment"],
        outputCol="features"
    )
    
    data = assembler.transform(df).select("features", "target")
    
    # Split data: 80% train, 20% test
    train, test = data.randomSplit([0.8, 0.2], seed=42)
    
    print(f"\nTrain set: {train.count()}, Test set: {test.count()}")
    
    # Train linear regression model
    lr = LinearRegression(
        featuresCol="features",
        labelCol="target",
        maxIter=20,
        regParam=0.1
    )
    
    model = lr.fit(train)
    
    # Evaluate model
    predictions = model.transform(test)
    
    evaluator = RegressionEvaluator(
        predictionCol="prediction",
        labelCol="target",
        metricName="rmse"
    )
    
    rmse = evaluator.evaluate(predictions)
    
    r2_evaluator = RegressionEvaluator(
        predictionCol="prediction",
        labelCol="target",
        metricName="r2"
    )
    
    r2 = r2_evaluator.evaluate(predictions)
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f} units")
    print(f"R² (Coefficient of Determination): {r2:.4f}")
    print(f"Model Coefficients: {model.coefficients.toArray()}")
    print(f"Intercept: {model.intercept:.2f}")
    
    print("\nSample Predictions (Actual vs Predicted):")
    predictions.select("target", "prediction").show(15)
    
    return model, predictions

def make_forecast(spark, predictions):
    """Generate summary forecast statistics"""
    
    print("\n" + "="*60)
    print("FORECAST SUMMARY")
    print("="*60)
    
    # Calculate prediction error
    from pyspark.sql.functions import abs, avg
    
    error_df = predictions.withColumn(
        "error", 
        abs(col("target") - col("prediction"))
    ).withColumn(
        "error_pct",
        (abs(col("target") - col("prediction")) / col("target") * 100)
    )
    
    avg_error = error_df.agg(avg("error_pct")).collect()[0][0]
    
    print(f"\nAverage Forecast Error: {avg_error:.2f}%")
    print("\nThis model can now forecast demand for:")
    print("- Next week product demand")
    print("- Inventory optimization")
    print("- Revenue forecasting")
    print("\nIn production, this would integrate with:")
    print("- Supply chain planning systems")
    print("- Inventory management")
    print("- Financial forecasting")

def main():
    spark = create_spark_session()
    
    try:
        model, predictions = train_forecast_model(spark)
        make_forecast(spark, predictions)
        
        print("\n" + "="*60)
        print("MODEL TRAINING COMPLETE")
        print("="*60)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()