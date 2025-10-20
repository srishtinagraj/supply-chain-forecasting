from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, count, date_format, lag, year, month, dayofmonth
from pyspark.sql.window import Window
import os

def create_spark_session():
    """Initialize Spark - simple, no Iceberg extensions"""
    spark = SparkSession.builder \
        .appName("apple-supply-chain-forecasting") \
        .master("local[*]") \
        .getOrCreate()
    return spark

def load_and_process_data(spark):
    """Load CSV data and perform initial transformations"""
    df = spark.read.csv('data/sales_data.csv', header=True, inferSchema=True)
    
    print("\n" + "="*60)
    print("RAW DATA PREVIEW")
    print("="*60)
    print(f"Total records: {df.count()}")
    df.show(5)
    
    return df

def save_to_parquet(df, path):
    """Save to Parquet format (efficient columnar storage)"""
    df.write.mode("overwrite").parquet(path)
    print(f" Saved to {path}")

def compute_daily_metrics(spark, df):
    """Compute daily aggregated metrics - this is where the analytics happens"""
    
    # Add date components
    df_dated = df.withColumn(
        "date_only", 
        date_format(col("date"), "yyyy-MM-dd")
    )
    
    # Daily aggregations by product
    daily_metrics = df_dated.groupBy(
        col("date_only").alias("date"),
        col("product")
    ).agg(
        sum("units_sold").alias("total_units"),
        sum("revenue").alias("total_revenue"),
        avg("price").alias("avg_price"),
        avg("social_sentiment").alias("avg_sentiment"),
        count("*").alias("transaction_count")
    ).orderBy("date", "product")
    
    print("\n" + "="*60)
    print("DAILY METRICS (aggregated by product)")
    print("="*60)
    daily_metrics.show(10)
    
    # Save metrics
    save_to_parquet(daily_metrics, "output/daily_metrics")
    
    return daily_metrics

def compute_regional_analysis(spark, df):
    """Analyze sales by region"""
    regional = df.groupBy("region", "product").agg(
        sum("units_sold").alias("total_units"),
        sum("revenue").alias("total_revenue"),
        avg("social_sentiment").alias("avg_sentiment")
    ).orderBy("total_revenue", ascending=False)
    
    print("\n" + "="*60)
    print("REGIONAL ANALYSIS (top performing regions)")
    print("="*60)
    regional.show(10)
    
    save_to_parquet(regional, "output/regional_analysis")
    return regional

def compute_growth_metrics(spark, daily_metrics):
    """Calculate day-over-day growth rates"""
    
    window_spec = Window.partitionBy("product").orderBy("date")
    
    growth_data = daily_metrics.withColumn(
        "prev_day_units", 
        lag("total_units").over(window_spec)
    ).withColumn(
        "units_growth_pct",
        ((col("total_units") - col("prev_day_units")) / col("prev_day_units") * 100)
    ).filter(col("prev_day_units").isNotNull())
    
    print("\n" + "="*60)
    print("GROWTH ANALYSIS (day-over-day trends)")
    print("="*60)
    growth_data.select("date", "product", "total_units", "units_growth_pct").show(15)
    
    save_to_parquet(growth_data, "output/growth_metrics")
    return growth_data

def compute_forecast_features(spark, growth_data):
    """Prepare features for forecasting model"""
    
    # Select and rename for clarity
    features = growth_data.select(
        col("product"),
        col("date"),
        col("total_units").alias("target"),
        col("prev_day_units").alias("prev_units"),
        col("units_growth_pct").alias("growth_rate"),
        col("avg_sentiment").alias("sentiment")
    )
    
    print("\n" + "="*60)
    print("FORECASTING FEATURES (ready for ML)")
    print("="*60)
    features.show(10)
    
    save_to_parquet(features, "output/forecast_features")
    return features

def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    print("\n🚀 APPLE SUPPLY CHAIN ANALYTICS PIPELINE")
    print("Using Apache Spark for distributed processing\n")
    
    # Step 1: Load data
    df = load_and_process_data(spark)
    
    # Step 2: Compute daily metrics
    daily_metrics = compute_daily_metrics(spark, df)
    
    # Step 3: Regional analysis
    compute_regional_analysis(spark, df)
    
    # Step 4: Growth metrics
    growth_metrics = compute_growth_metrics(spark, daily_metrics)
    
    # Step 5: Prepare features for forecasting
    compute_forecast_features(spark, growth_metrics)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print("Output saved to /output directory:")
    print("  - daily_metrics")
    print("  - regional_analysis")
    print("  - growth_metrics")
    print("  - forecast_features")
    print("\nReady for forecasting model training!")

if __name__ == "__main__":
    main()