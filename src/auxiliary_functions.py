

def plot_continuous_variable(df, col_name, target=None):
    """
    Plot continuous variables, with or without the target variable.

    Parameters:
    - df: pd.DataFrame
    DataFrame containing the data.
    - col_name: str
    Name of the column to visualize.
    - target: str, optional
    Target variable for facets (default is None).
    """

    count_null = df[col_name].isnull().sum()  # Count the null values in a specific column

    fig, axes = plt.subplots(1, 2 if target else 1, figsize=(14, 5), dpi=90)  # Create subplots

    # Plot a histogram with KDE (Kernel Density Estimation) for the continuous variable
    sns.histplot(df[col_name].dropna(), kde=True, ax=axes[0] if target else axes, color="skyblue")
    axes[0 if target else 0].set_title(f"{col_name} (nulls: {count_null})")
    axes[0 if target else 0].set_xlabel(col_name)
    axes[0 if target else 0].set_ylabel("Count")

    if target:  # If a target variable is provided, we plot a boxplot
        sns.boxplot(x=target, y=col_name, data=df, ax=axes[1], palette="Set2")
        axes[1].set_title(f"{col_name} by {target}")
        axes[1].set_ylabel(col_name)
        axes[1].set_xlabel(target)

    plt.tight_layout()  # Adjust the spacing
    plt.show()  # Display the plot



def plot_categorical_variable(df, col_name, target=None):
    """
    Plot categorical variables, with or without the target variable.

    Parameters:
    - df: pd.DataFrame
        DataFrame containing the data.
    - col_name: str
        Name of the column to visualize.
    - target: str, optional
        Target variable for facets (default is None).
    """

    count_null = df[col_name].isnull().sum()  # Count the null values in the column

    # Handle too many categories (limit to the 10 most frequent)
    unique_vals = df[col_name].astype(str).value_counts()  # Count the unique values
    if len(unique_vals) > 10:  # If there are more than 10 categories, we limit to the top 10
        top_vals = unique_vals.head(10).index
        df = df[df[col_name].astype(str).isin(top_vals)]

    fig, axes = plt.subplots(1, 2 if target else 1, figsize=(14, 5), dpi=90)  # Create subplots

    # Plot a countplot for the categorical variable
    sns.countplot(
        x=df[col_name].astype(str),
        order=sorted(df[col_name].astype(str).unique()),
        ax=axes[0] if target else axes,
        color="skyblue"
    )
    axes[0 if target else 0].set_title(f"{col_name} (nulls: {count_null})")
    axes[0 if target else 0].set_xlabel(col_name)
    axes[0 if target else 0].set_ylabel("Count")
    axes[0 if target else 0].tick_params(axis='x', rotation=45)  # Rotate x-axis labels

    if target:  #If a target variable is provided, we plot the proportions of each class
        proportions = (
            df.groupby(col_name)[target]  # Group by the categorical variable
            .value_counts(normalize=True)  # Plot a countplot for the categorical variable
            .rename("proportion")
            .reset_index()
        )
        sns.barplot(
            x=col_name,
            y="proportion",
            hue=target,
            data=proportions,
            ax=axes[1],
            palette="Set2"
        )
        axes[1].set_title(f"Proportions of {target} by {col_name}")
        axes[1].set_xlabel(col_name)
        axes[1].set_ylabel("Proportion")
        axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()  # Adjust the spacing
    plt.show()  # Display the plot



def plot_feature_variable(df, col_name, isContinuous, target=None):
    """
    Decide which plotting function to call based on the variable type (continuous or categorical).

    Parameters:
    - df: pd.DataFrame
        DataFrame containing the data.
    - col_name: str
        Name of the column to visualize.
    - isContinuous: bool
        Whether the variable is continuous or categorical.
    - target: str, optional
        Target variable for facets (default is None).
    """

    if isContinuous:
        plot_continuous_variable(df, col_name, target=target)  # If it is continuous, call the function for continuous variables
    else:
        plot_categorical_variable(df, col_name, target=target)  # If it is categorical, call the function for categorical variables



def analyze_outliers(credit_processed, list_var_num, target, multiplier=3):
    """
    Analyze the outliers for numerical variables and return the results.
    - df: DataFrame
    - list_var_num: list of numerical variables to analyze
    - target: target variable for analysis
    - multiplier: factor to detect outliers (default is 3 times the IQR)
    """

    outliers = []
    for col in list_var_num:
        q1 = credit_processed[col].quantile(0.25)
        q3 = credit_processed[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        # Identify the outliers in the column
        outliers_in_col = credit_processed[(credit_processed[col] < lower_bound) | (credit_processed[col] > upper_bound)]
        outliers.append((col, len(outliers_in_col)))

    # Create a DataFrame with the outlier results
    outlier_df = pd.DataFrame(outliers, columns=["Variable", "Outlier Count"])
    return outlier_df



def treat_outliers(credit_processed, list_var_num, multiplier=3, method="median"):
    """
    Handle the outliers in numerical variables by replacing them with the mean or median.

    - df: DataFrame
    - list_var_num: list of numerical variables to treat
    - multiplier: factor to detect outliers (default is 3 times the IQR)
    - method: 'median' or 'mean' to indicate which method to use for replacing the outliers
    """

    for col in list_var_num:
        q1 = credit_processed[col].quantile(0.25)
        q3 = credit_processed[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        outliers_in_col = (credit_processed[col] < lower_bound) | (credit_processed[col] > upper_bound)

        if method == "median":
            median_value = credit_processed[col].median()
            credit_processed[col] = credit_processed[col].where(~outliers_in_col, median_value)

        elif method == "mean":
            mean_value = credit_processed[col].mean()
            credit_processed[col] = credit_processed[col].where(~outliers_in_col, mean_value)

    return credit_processed



def analyze_nulls(df):
    """
    Analyze the null values in a DataFrame, both by columns and by rows.

    - df: DataFrame to analyze.
    """


    nulls_columns = df.isnull().sum()
    percentage_nulls_columns = (nulls_columns / len(df) * 100).round(2)

    pd_null_columns = pd.DataFrame({
        'nulls_columns': nulls_columns,
        'percentage (%)': percentage_nulls_columns
    }).sort_values(by='nulls_columns', ascending=False)

    nulls_rows = df.isnull().sum(axis=1)
    percentage_nulls_rows = (nulls_rows / df.shape[1] * 100).round(2)

    pd_null_rows = pd.DataFrame({
        'nulls_rows': nulls_rows,
        'percentage (%)': percentage_nulls_rows
    }).sort_values(by='nulls_rows', ascending=False)

    return pd_null_columns, pd_null_rows



def calculate_nulls_by_objective(df, target_col, type_variable):
    """
    Calculate the percentage of null values per column, grouped by the target variable (TARGET).

    - df: DataFrame containing the data.
    - target_col: Name of the target column (TARGET).
    - type_variable: 'categorical' or 'continuous' to select the type of variables to analyze.
    """


    if type_variable == 'categorical':
        columns = list_var_cat
    elif type_variable == 'continuous':
        columns = list_var_continuous
    else:
        raise ValueError("The type_variable must be 'categorical' or 'continuous'.")

    columns = [col for col in columns if col in df.columns]

    nulls_by_objective = pd.DataFrame(index=columns)

    grouped = df.groupby(target_col)
    for target_value, group in grouped:
        nulls_by_objective[f"Target_{int(target_value)}"] = (
            group[columns].isnull().sum() / len(group) * 100
        ).round(2)

    nulls_by_objective["Total_Percentage (%)"] = nulls_by_objective.mean(axis=1).round(2)

    if "Target_1" in nulls_by_objective.columns and "Target_0" in nulls_by_objective.columns:
        nulls_by_objective["Difference_0_1 (%)"] = (
            nulls_by_objective["Target_1"] - nulls_by_objective["Target_0"]
        ).round(2)

    return nulls_by_objective.sort_values(by="Total_Percentage (%)", ascending=False)



def get_corr_matrix(dataset, method='pearson', size_figure=[10, 8]):

    """
    Create a correlation matrix for the numerical variables in the dataset and visualize it.

    - dataset: DataFrame containing the data.
    - method: The correlation method, default is 'pearson'.
    - size_figure: Figure size, default is [10, 8].
    """


    if dataset is None:
        print('Arguments are missing for the function')
        return None
    corr = dataset.corr(method=method)
    for i in range(corr.shape[0]):
        corr.iloc[i, i] = 0
    plt.figure(figsize=size_figure)
    sns.heatmap(corr, center=0, square=True, linewidths=.5, cmap='viridis')
    plt.show()
    return corr



def cramers_v(confusion_matrix):
    """
    Calculate the value of Cramér's V, a measure of association between two categorical variables.

    - confusion_matrix: contingency matrix (frequency table) between two categorical variables.
    """


    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

