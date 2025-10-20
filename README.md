cat > README.md << 'EOF'
# Apple Supply Chain Demand Forecasting

A data engineering project demonstrating demand forecasting for Apple products using Apache Spark and Iceberg, directly relevant to Apple's supply chain optimization challenges.

## Project Overview

This pipeline simulates Apple's demand forecasting workflow:
1. **Data Generation**: Simulates 1 year of product sales data across regions
2. **Data Processing**: Spark SQL aggregations for daily metrics
3. **Iceberg Integration**: ACID-compliant data storage with time travel capability
4. **Forecasting**: Linear regression model for demand prediction

## Technologies

- **Apache Spark**: Distributed data processing
- **Apache Iceberg**: Versioned data lakehouse format with ACID guarantees
- **Scikit-learn/Spark ML**: Demand forecasting model
- **Python**: Data pipeline orchestration

## Why This Matters for Apple

- Supply chain optimization directly impacts margins
- Accurate demand forecasting prevents overstock/understock costs
- Iceberg's ACID guarantees ensure data reliability at scale
- Time travel enables recovery from data quality issues

## Project Structure