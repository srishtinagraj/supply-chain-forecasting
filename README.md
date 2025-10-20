cat > README.md << 'EOF'
# Apple Supply Chain Demand Forecasting

A data engineering project demonstrating demand forecasting for Apple products using Apache Spark, directly relevant to Apple's supply chain optimization challenges.

## Project Overview

This pipeline simulates Apple's demand forecasting workflow:
1. **Data Generation**: Simulates 1 year of product sales data across regions
2. **Data Processing**: Spark SQL aggregations for daily metrics
3. **Forecasting**: Linear regression model for demand prediction

## Technologies

- **Apache Spark**: Distributed data processing
- **Scikit-learn/Spark ML**: Demand forecasting model
- **Python**: Data pipeline orchestration

## Why This Matters for Apple

- Supply chain optimization directly impacts margins
- Accurate demand forecasting prevents overstock/understock costs
- Time travel enables recovery from data quality issues

## Model Performance

**Results:**
- RMSE: 1019.95 units
- R²: 0.4585 (explains 45% of variance)
- Average Forecast Error: 35.42%

**Why this is reasonable:**
The model uses only 3 features on synthetic data with limited patterns. In production, forecasting accuracy improves with:
- Real historical data with true seasonality
- More features (marketing, competitor data, inventory levels)
- Advanced models (Prophet, XGBoost)
- Time-series specific techniques

**Key Insight:**
The coefficients show that growth_rate (8.92) is much more predictive than previous_units (0.53). This means Apple should focus on momentum, not just absolute numbers.