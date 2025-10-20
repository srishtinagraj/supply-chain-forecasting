from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, count, date_format, lag
from pyspark.sql.window import Window
import os

def create_spark_session():
    """Initialize Spark with Iceberg support"""
    spark = SparkSession.builder \
        .appName("apple-supply-chain-forecasting") \
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
        .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
        .config("spark.sql.catalog.local.type", "hadoop") \
        .config("spark.sql.catalog.local.warehouse", "./spark-warehouse") \
        .getOrCreate()
    
    return spark

def load_and_process_data(spark):
    """Load CSV data and perform initial transformations"""
    df = spark.read.csv('data/sales_data.csv', header=True, inferSchema=True)
    
    print("Raw data shape:", df.count())
    df.show(5)
    
    return df

def create_iceberg_table(spark, df):
    """Write data to Iceberg table for versioning and ACID guarantees"""
    # Remove existing table if it exists
    spark.sql("DROP TABLE IF EXISTS local.default.apple_sales")
    
    # Write to Iceberg
    df.write \
        .format("iceberg") \
        .mode("overwrite") \
        .saveAsTable("local.default.apple_sales")
    
    print("Iceberg table created: local.default.apple_sales")

def compute_daily_metrics(spark):
    """Compute daily aggregated metrics for forecasting"""
    df = spark.table("local.default.apple_sales")
    
    # Daily aggregations
    daily_metrics = df.groupBy(
        date_format(col("date"), "yyyy-MM-dd").alias("date"),
        col("product")
    ).agg(
        sum("units_sold").alias("total_units"),
        sum("revenue").alias("total_revenue"),
        avg("price").alias("avg_price"),
        avg("social_sentiment").alias("avg_sentiment"),
        count("*").alias("transaction_count")
    ).orderBy("date", "product")
    
    # Write daily metrics to Iceberg
    spark.sql("DROP TABLE IF EXISTS local.default.daily_metrics")
    daily_metrics.write \
        .format("iceberg") \
        .mode("overwrite") \
        .saveAsTable("local.default.daily_metrics")
    
    print("Daily metrics computed and stored in Iceberg")
    daily_metrics.show(10)
    
    return daily_metrics

def demonstrate_time_travel(spark):
    """Show Iceberg's time travel capability"""
    print("\n=== DEMONSTRATING ICEBERG TIME TRAVEL ===")
    
    # Get current snapshot
    current = spark.sql("SELECT * FROM local.default.daily_metrics LIMIT 5")
    print("Current data:")
    current.show()
    
    # This shows how you could query historical data
    print("Time travel queries available in production with Iceberg")
    print("Example: SELECT * FROM local.default.daily_metrics VERSION AS OF '2024-10-15'")

def compute_forecast_features(spark):
    """Create features for demand forecasting"""
    df = spark.table("local.default.daily_metrics")
    
    window_spec = Window.partitionBy("product").orderBy("date")
    
    forecast_data = df.withColumn(
        "prev_day_units", lag("total_units").over(window_spec)
    ).withColumn(
        "units_growth_pct", 
        ((col("total_units") - col("prev_day_units")) / col("prev_day_units") * 100)
    ).filter(col("prev_day_units").isNotNull())
    
    # Write forecasting features to Iceberg
    spark.sql("DROP TABLE IF EXISTS local.default.forecast_features")
    forecast_data.write \
        .format("iceberg") \
        .mode("overwrite") \
        .saveAsTable("local.default.forecast_features")
    
    print("Forecast features computed")
    forecast_data.select("date", "product", "total_units", "units_growth_pct").show(10)
    
    return forecast_data

def main():
    spark = create_spark_session()
    
    # Step 1: Load data
    df = load_and_process_data(spark)
    
    # Step 2: Create Iceberg table
    create_iceberg_table(spark, df)
    
    # Step 3: Compute metrics
    compute_daily_metrics(spark)
    
    # Step 4: Show time travel capability
    demonstrate_time_travel(spark)
    
    # Step 5: Create forecasting features
    compute_forecast_features(spark)
    
    print("\n✅ Pipeline complete! Data ready for forecasting")

if __name__ == "__main__":
    main()