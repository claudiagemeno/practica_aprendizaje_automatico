# practica_aprendizaje_automatico

## **General Description**

This project addresses the challenge of working with highly imbalanced data, from initial exploration to the implementation of predictive models. Five notebooks were developed to document a comprehensive workflow, including data cleaning, feature selection, class balancing, modeling, and interpretability analysis. The approach combines statistical techniques and machine learning to optimize prediction in an imbalanced environment, prioritizing accuracy and model understanding.

## **Project Contents**

### Notebook 1: Initial Exploration and Analysis
This first notebook performs a detailed exploratory analysis to understand the characteristics of the dataset. Missing values, outliers, and relationships between variables were identified. The correlation between continuous and categorical variables with the target variable was also assessed.
The analysis revealed a marked imbalance in the target variable’s classes, guiding subsequent decisions on balancing and modeling. This notebook laid the foundation for identifying key variables and potential issues.

### Notebook 2: Handling Missing Values, Outliers, and Correlations
This notebook focuses on data cleaning and preprocessing. Missing values were imputed with medians and class-specific strategies for each target class. Outliers were handled using the interquartile range (IQR), and the dataset was scaled to normalize numerical variables.
Correlations (Pearson and Cramér’s V) were analyzed to select relevant variables. This allowed for noise reduction and established a solid foundation for feature selection and modeling.

### Notebook 3: Encoding, Scaling, and Feature Selection
This notebook applies advanced encoding techniques to transform categorical variables (One-Hot Encoding, Target Encoding, and binary mappings) and scales numerical variables using StandardScaler. Feature selection combined methods like Lasso, Ridge, and Decision Tree, integrating results to ensure selected variables maximized model performance.
This balanced approach allowed the selection of the most relevant features, though metrics still reflected the challenges of working with imbalanced data.

### Notebook 4: Predictive Modeling
Here, various classification models were trained and evaluated, including Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, and SVM. Class balancing techniques like SMOTE were used to address class imbalance, and hyperparameters were optimized using Grid Search and Randomized Search.
The Generalized Linear Model (GLM) stood out for its balance between accuracy and simplicity, showing consistent metrics for both classes, although the minority class remained a challenge in terms of accuracy and F1-Score.

### Notebook 5: Model Interpretability
This notebook was dedicated to interpreting the final model using SHAP (SHapley Additive exPlanations). As the analysis involved dimensionality reduction with PCA, the explanations focused on the principal components (PCs) and their relationship to the most influential original variables.
Explanatory plots, such as summary plots, waterfall plots, and force plots, were generated to show how variables contribute positively or negatively to model predictions, improving transparency and understanding of the final model.


## **Technologies and Tools Used**
**Python Libraries:** pandas, numpy, scikit-learn, imbalanced-learn, shap, xgboost, lightgbm, catboost
**Visualization:** matplotlib, seaborn, shap
**Class Balancing:** SMOTE, Under-sampling
**Feature Selection:** Lasso, Ridge, Decision Tree, PCA
**Modeling**: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, GLM, SVM

## **General Conclusion**
This project demonstrates a comprehensive approach to handling imbalanced data. While the metrics reflect the inherent difficulties of predicting the minority class, a robust and balanced model was achieved through a strategic combination of class balancing, feature selection, and modeling techniques.
The GLM model was chosen as the final solution due to its consistent performance and interpretability. Despite the challenges posed by the extreme class imbalance, the project highlights the ability to optimize both accuracy and interpretability, providing a reliable and transparent solution for binary classification tasks.

## **Contributors**
Lucía Poyán, Nuria Olmedilla, Claudia Gemeno
**Github**: https://github.com/claudiagemeno/practica_aprendizaje_automatico.git 
