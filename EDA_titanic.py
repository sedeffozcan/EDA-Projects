#Advanced Functional Exploratory Data Analysis

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df = sns.load_dataset("titanic")
df.head() # the first 5 observation units
df.tail() # the last 5 observation units
df.shape  # the row and column numbers
df.info() # variables, missing values, types of variables
df.dtypes # types of the variables
df.columns # names of variables
df.index # index information
df.isnull().values.any() # checks whether there are any missing values
df.isnull().sum() # missing values of variable
df.describe().T # some statistics such as mean, count, sum, etc.


def check_df(dataframe, head=5,tail=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(tail))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0.05,0.25, 0.50,0.75, 0.95, 0.99]).T)
    print("###################### Unique Values #################")
    print(dataframe.nunique())

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

cat_cols = cat_cols + num_but_cat # if no variable in cat_but_car
cat_cols = [col for col in cat_cols if col not in cat_but_car] # if there is a variable in cat_but_car

df[cat_cols]
df[cat_cols].nunique() # number of classes in all variables

cat_cols = [col for col in cat_cols if col not in cat_cols]

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name:
                  dataframe[col_name].value_counts(),
                        "Ratio": 100 *
        dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    if df[col].dtypes == "bool":
        print("sdfsdfsdfsdfsdfsd")
    else:
        cat_summary(df, col, plot=True)

def cat_summary(dataframe, col_name, plot=False):

    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)

        print(pd.DataFrame({col_name:
         dataframe[col_name].value_counts(),
                            "Ratio": 100 *
         dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)
    else:
        print(pd.DataFrame({col_name:
        dataframe[col_name].value_counts(),
                            "Ratio": 100 *
        dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

num_cols = [col for col in df.columns if df[col].dtypes in ["int","float"]]
num_cols = [col for col in num_cols if col not in cat_cols]

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70,
    0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col)

def grab_col_names(dataframe, cat_th=10,  car_th=20):
    """
    It gives the names of categorical, numerical, and categorical
    but cardinal variables.

    Parameters
    ----------
    dataframe: dataframe
        The data frame from which we want to take variables
    cat_th: int, float
        The threshold for the number of classes of numerical but
        categorical variables
    car_th: int, float
        The threshold for the number of classes of cardinal but
        categorical variables
    Returns
    -------
    cat_cols: list
        List of categorical variables
    num_cols: list
        List of numerical variables
    cat_but_car: list
        List of cardinal but categorical variables
    Notes
    ------
    cat_cols + num_cols + cat_but_car = total number of variables
    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns
    if   str(dataframe[col].dtypes) in ["category", "object",
    "bool"]]
    num_but_cat = [col for col in dataframe.columns if
    dataframe[col].nunique() < 10 and dataframe[col].dtypes in
    ["int", "float"]]
    cat_but_car = [col for col in dataframe.columns if
                    dataframe[col].nunique() > 20 and
    str(dataframe[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if
    dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN":
    dataframe.groupby(categorical_col)[target].mean()}),
    end="\n\n\n")

# apply this function to all categorical variables
for col in cat_cols:
    target_summary_with_cat(df, "survived", col)
Analysis of Target Variable with Numerical Variables
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}),
    end="\n\n\n")

# apply this function to all numerical variables
for col in num_cols:
    target_summary_with_num(df, "survived", col)

# Correlation
num_cols = [col for col in df.columns if df[col].dtype in [int, float]]
corr = df[num_cols].corr()
To draw heat map:
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

cor_matrix = df.corr().abs()

upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))

drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>0.90) ]
cor_matrix[drop_list]
df.drop(drop_list, axis=1)

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix =
    cor_matrix.where(np.triu(np.ones(cor_matrix.shape),
    k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if
    any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list
# Apply the function to the dataframe df
high_correlated_cols(df)
drop_list = high_correlated_cols(df, plot=True)
df.drop(drop_list, axis=1)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)