
# ADVANCED FUNCTIONAL EXPLORATORY DATA ANALYSIS

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

df_=pd.read_csv("/Users/sedeftaskin/Desktop/Data_Science/archive/video_games_sales.csv")
df=df_.copy()


#Name: Name of the game
#Platform: Console on which the game is running
#Year_of_Release: Year of the game released
#Genre: Game’s category
#Publisher: Publisher
#NA_Sales: Game sales in North America (in millions of units)
#EU_Sales: Game sales in the European Union (in millions of units)
#JP_Sales: Game sales in Japan (in millions of units)
#Other_Sales: Game sales in the rest of the world, i.e. Africa, Asia excluding Japan, Australia, Europe excluding the E.U. and South America (in millions of units)
#Global_Sales: Total sales in the world (in millions of units)
#Critic_Score: Aggregate score compiled by Meta critic staff
#Critic_Count: The number of critics used in coming up with the Critic_score
#User_Score: Score by Metacritic’s subscribers
#User_Count: Number of users who gave the user_score
#Developer: Party responsible for creating the game
#Rating: The ESRB ratings: E for “Everyone”; E10+ for “Everyone 10+”; T for “Teen”; M for “Mature”; AO for “Adults Only”; RP for

#Motivation
#You are working as a data analyst for a video game retailer based in Japan. The retailer typically orders games based on sales in North America and Europe,
# as the games are often released later in Japan.
# However, they have found that North American and European sales are not always a perfect predictor of how a game will sell in Japan.
# Your manager has asked you to develop a model that can predict the sales in Japan using sales in North America and Europe and other attributes
# such as the name of the game, the platform, the genre, and the publisher.
# You will need to prepare a report that is accessible to a broad audience. It should outline your motivation, steps, findings, and conclusions.


# By multiplying 1000000 we get the actual sale, (target variables)


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

check_df(df)


# There are lots of null values in Critic_Score,....
# So we remove them
for col in df.columns:
    if df[col].isnull().sum() > len(df)/4:
        df.drop(col,axis=1,inplace=True)
# We remove the null values
df.dropna(inplace=True)

for col in df.columns:
    if "Sales" in col:
        df[col]=(df[col]*1000000).astype("int")

# how to manage it?

df[df["Global_Sales"]==(df["NA_Sales"]+df["EU_Sales"]+df["JP_Sales"]+df["Other_Sales"])]
df[df["Global_Sales"]!=(df["NA_Sales"]+df["EU_Sales"]+df["JP_Sales"]+df["Other_Sales"])]
df[df["Global_Sales"] > (df["NA_Sales"]+df["EU_Sales"]+df["JP_Sales"]+df["Other_Sales"])+20000]
df[df["Global_Sales"] < (df["NA_Sales"]+df["EU_Sales"]+df["JP_Sales"]+df["Other_Sales"])-20000]

# we assign Global_Sales as sum of all Sale variables

df["Global_Sales"]=df["NA_Sales"]+df["EU_Sales"]+df["JP_Sales"]+df["Other_Sales"]


# We convert the type of the year variable into int
df["Year_of_Release"]=df["Year_of_Release"].astype("int")


# Now there are 10 variables 16416 rows...
df.shape
df.columns

def grab_col_names(dataframe, cat_th=10,  car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df,40,60)


#analyzing of categorical variables
def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))


for col in cat_cols:
    cat_summary(df, col)

def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

for col in num_cols:
    num_summary(df, col)

# Simply visualization of Categorical Variables

for col in cat_cols:
    df[col].value_counts().plot(kind='bar')
    plt.xlabel(col)
    plt.show(block=True)

#correlation

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc={'figure.figsize': (8, 8)})
        sns.heatmap(corr,cmap="flare",annot=True)
        plt.show()
    return drop_list

high_correlated_cols(df)
drop_list = high_correlated_cols(df, plot=True)
# We won't remove any Sale variable.

# Global Sale and Year Relation
df_global_year=df.groupby("Year_of_Release").agg({"Global_Sales":["max","min","sum","mean"]})
df_global_year.reset_index(inplace=True)
# Q: What are the top 5 years of release that are making high sales globally?

df_global_year.sort_values(by=("Global_Sales","sum"),ascending=False).head()

# Regional (Japan) Sale and Year Relation
df_JP_year=df.groupby("Year_of_Release").agg({"JP_Sales":["max","min","sum","mean"]})
df_JP_year.reset_index(inplace=True)

# Q: What are the top 5 years of release that are making high sales in Japan?

df_JP_year.sort_values(by=("JP_Sales","sum"),ascending=False).head()

# Q: What are the top 10 games making the most sales globally, in EU, in NA, in JP?
df.groupby("Name").agg({"Global_Sales":"sum"}).sort_values("Global_Sales",ascending=False).head(10)
df.groupby("Name").agg({"NA_Sales":"sum"}).sort_values("NA_Sales",ascending=False).head(10)
df.groupby("Name").agg({"EU_Sales":"sum"}).sort_values("EU_Sales",ascending=False).head(10)
df.groupby("Name").agg({"JP_Sales":"sum"}).sort_values("JP_Sales",ascending=False).head(10)

# Q: What are the top 5 gaming Genres, Platforms that are making high sales gloabbly, in EU, in NA, in JP?
# Genre
df.groupby("Genre").agg({"Global_Sales":"sum"}).sort_values("Global_Sales",ascending=False).head()
df.groupby("Genre").agg({"NA_Sales":"sum"}).sort_values("NA_Sales",ascending=False).head()
df.groupby("Genre").agg({"JP_Sales":"sum"}).sort_values("JP_Sales",ascending=False).head()

# Platform
df.groupby("Platform").agg({"EU_Sales":"sum"}).sort_values("EU_Sales",ascending=False).head()
df.groupby("Platform").agg({"JP_Sales":"sum"}).sort_values("JP_Sales",ascending=False).head()
df.groupby("Platform").agg({"Global_Sales":"sum"}).sort_values("Global_Sales",ascending=False).head()

# Q: Which Publishers made the most sales globally , in EU, in NA, in JP?

df.groupby("Publisher").agg({"Global_Sales":"sum"}).sort_values("Global_Sales",ascending=False).head()
df.groupby("Publisher").agg({"NA_Sales":"sum"}).sort_values("NA_Sales",ascending=False).head()
df.groupby("Publisher").agg({"JP_Sales":"sum"}).sort_values("JP_Sales",ascending=False).head()

# Q: What Wii games were sold?

df_Wii=df[df["Platform"]=="Wii"]
df_Wii["JP_Sales"].sum()
df_Wii["Global_Sales"].sum()
df_Wii.Name.unique()[0:10]
df_Wii.Name.nunique()

# What PC-FX games were sold?

df[df["Platform"]=="PCFX"]


# What Misc games were sold?

df_Misc=df[df["Genre"]=="Misc"]
df_Misc.Name.unique() # games of kind Misc

#  Analyze: Are some genres significantly more likely to perform better or worse in Japan than others? If so, which ones?
