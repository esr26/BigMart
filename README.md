# BigMart Sales Prediction – Machine Learning Coding Test

## Project Overview

This project focuses on predicting product level sales across multiple BigMart outlets using historical sales data. The dataset includes both product-level attributes (price, category, visibility, weight) and outlet-level information (type, size, and location tier).

The objective was not only to build an accurate predictive model but also to understand the key factors influencing sales performance across different store formats and product categories.



## Dataset Information

- **Training Data:** 8,523 rows  
- **Test Data:** 5,681 rows  
- **Target Variable:** `Item_Outlet_Sales`  
- Features include both numerical and categorical attributes.


## Key Insights from Exploratory Data Analysis

During exploratory analysis, several meaningful patterns were observed:

- Three major product groups were identified from the Item Identifier and used to create a new feature: **Item_Category**.
- `Outlet_Type` showed a stronger influence on sales compared to `Outlet_Size`.
- **Supermarket Type3** had the highest median sales, while Grocery Stores consistently underperformed.
- Medium-sized outlets performed better than large outlets, suggesting that bigger size does not guarantee higher revenue.
- `Item_MRP` showed the strongest positive correlation (0.57) with sales.
- Non-Consumables and Snack Foods emerged as reliable high-performing categories.

These insights directly influenced the feature engineering and modeling strategy.



## Feature Engineering

Based on EDA findings, the following transformations were implemented:

- Created `Item_Category` from Item Identifier  
- Engineered `Item_Age` using establishment year  
- Corrected zero values in `Item_Visibility`  
- Created `Visibility_MeanRatio`  
- Introduced `MRP_Band` for price segmentation  
- Cleaned categorical inconsistencies  

All preprocessing logic is modularized in:



## Modeling Approach

Three machine learning models were evaluated using consistent 5-fold cross-validation:

- Random Forest  
- XGBoost  
- CatBoost  

Hyperparameter tuning was performed using RandomizedSearch and structured experimentation to ensure fair comparison.



## Final Model Selection

CatBoost achieved the best cross-validation performance.

| Model          | Best CV RMSE |
|---------------|-------------|
| **CatBoost**  | ~1075       |
| Random Forest | ~1095       |
| XGBoost       | ~1107       |

CatBoost was selected due to its strong handling of categorical variables and consistent performance across folds.



## Final Results

- **Cross-Validation RMSE:** ~1075  
- **Leaderboard RMSE:** ~1145  
- **Leaderboard Rank:** 227  



## Conclusion

The analysis indicates that pricing (`Item_MRP`) and outlet format (`Outlet_Type`) are stronger drivers of sales than raw numerical attributes. Careful feature engineering combined with structured experimentation resulted in a stable and high performing CatBoost model.

This project emphasizes the importance of understanding business patterns before model optimization.
