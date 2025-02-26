import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import os
from dotenv import load_dotenv
from functions.functions import *

load_dotenv()

path = os.getenv("DIRECTORY_PATH")

df = pd.read_csv(f"{path}heart.csv")

print("\nTABLE DESCRIPTION")
print(df.describe())

print("\COLUMNS")
print(df.columns)

print("\nFIRST DATA OF THE TABLE")
print(df.head())

print("\nTABLE INFORMATION")
print(df.info())

print("\nNUMBER OF NULL VALUES PER COLUMN")
print(df.isnull().sum())

print("\nDESCRIPTION OF COLUMNS age, sex and cp")
print(df[['age', 'sex', 'cp']].describe())

print("\nDESCRIPTION OF COLUMNS trestbps, chol Y fbs")
print(df[['trestbps', 'chol', 'fbs']].describe())

print("\nDESCRIPTION OF COLUMNS restecg, thalach Y exang")
print(df[['restecg', 'thalach', 'exang']].describe())

print("\nDESCRIPTION OF COLUMNS oldpeak, slope Y ca")
print(df[['oldpeak', 'slope', 'ca']].describe())

df["ca"] = df["ca"].replace(4, np.nan)
sns.heatmap(df.isnull(), cbar=False)
plt.show()

df = drop_na(df)

print("\nDESCRIPTION OF COLUMN ca WITHOUT 4")
print(df["ca"].describe())

print("\nDESCRIPTION OF COLUMN thal y target")
print(df[['thal', 'target']].describe())

df["thal"] = df["thal"].replace(0, np.nan)
sns.heatmap(df.isnull(), cbar=False)
plt.show()

df = drop_na(df)

print("\nDESCRIPTION OF COLUMN thal y target")
print(df[['thal', 'target']].describe())

fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(9, 5))
axes = axes.flat

columnas_numeric = df.select_dtypes(include=['float64', 'int64']).columns

for i, colum in enumerate(columnas_numeric):
    sns.histplot(
        data=df,
        x=colum,
        stat="count",
        kde=True,
        color=(list(plt.rcParams['axes.prop_cycle']) * 2)[i]["color"],
        line_kws={'linewidth': 2},
        alpha=0.3,
        ax=axes[i]
    )
    axes[i].set_title(colum, fontsize=10, fontweight="bold")
    axes[i].tick_params(labelsize=8)
    axes[i].set_xlabel("")

fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Distribution of numerical variables', fontsize=10, fontweight="bold")
plt.show()

fig, axs = plt.subplots(nrows=2, ncols=7, figsize=(12, 6))
columnas_numeric = df.select_dtypes(include=['float64', 'int64']).columns

axs = axs.flatten()

for i, colum in enumerate(columnas_numeric):
    axs[i].hist(df[colum], bins=20, color="#3182bd", alpha=0.5)
    axs[i].plot(df[colum], np.full_like(df[colum], -0.01), '|k', markeredgewidth=1)
    axs[i].set_title(f'{colum}')
    axs[i].set_xlabel(colum)
    axs[i].set_ylabel('counts')

plt.tight_layout()
plt.show()


columnas_numeric = df.select_dtypes(include=['float64', 'int64']).columns
for i, colum in enumerate(columnas_numeric):
    print(f"\nCENTRALIZATION MEASURES {colum}")
    print(f'Mean:{df[colum].mean()} \
     \nMedian: {df[colum].median()} \
     \nMode: {df[colum].mode()}')

print(f'\nThe variance is:\n{df.var()}')

print(f'\nStandard Deviation per row:\n{df.std(axis=0)}')

for i, colum in enumerate(columnas_numeric):
    print(f"\nRANGE OF {colum} is: {df[colum].max() - df[colum].min()}")

for i, colum in enumerate(columnas_numeric):
    print(f"\nTHE IQR OF {colum} is: {df[colum].quantile(0.75) - df[colum].quantile(0.25)}")

cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100
print(
    f'\nThe coefficient of variation is:\n{df.select_dtypes(include=["int16", "int32", "int64", "float16", "float32", "float64"]).apply(cv)}')



print(f"\nAsymmetry measures are:\n{df.skew()}")

print(f"\nThe kurtosis measures are:\n{df.kurt()}")

df.plot(kind='box', subplots=True, layout=(2, 7),
        sharex=False, sharey=False, figsize=(20, 10))
plt.show()


df_without_outlier = delete_outliers(100, df, columnas_numeric)

print("\nTABLA SIN LOS OUTLIERS")
print(df_without_outlier.shape)
print(df_without_outlier.info())

df_without_outlier.plot(kind='box', subplots=True, layout=(2, 7),
                    sharex=False, sharey=False, figsize=(20, 10))
plt.show()

df_categoric = df_without_outlier.copy()

df_categoric['target'] = df_categoric.target.replace({1: "With_disease", 0: "Without_disease"})
df_categoric['sex'] = df_categoric.sex.replace({1: "Male", 0: "Female"})
df_categoric['cp'] = df_categoric.cp.replace(
    {0: "Typical_Angina", 1: "Atypic_Angina", 2: "Non-anginous_pain", 3: "Asymptomatic"})
df_categoric['exang'] = df_categoric.exang.replace({1: "No", 0: "Yes"})
df_categoric['fbs'] = df_categoric.fbs.replace({0: "True", 1: "False"})
df_categoric['slope'] = df_categoric.slope.replace({0: "Increasing", 1: "Flat", 2: "Decreasing"})
df_categoric['thal'] = df_categoric.thal.replace({1: "Reversible_defect", 2: "Fixed_defect ", 3: "Normal"})

categorical_columns = ['target', 'sex', 'cp', 'exang', 'fbs', 'slope', 'thal']

for col in categorical_columns:
    counts = df_categoric[col].value_counts()
    percentages = df_categoric[col].value_counts(normalize=True) * 100
    summary = pd.DataFrame({'Count': counts, 'Percentage': percentages})
    
    print(f"\nColumn: {col}")
    print(summary)




fig, ax = plt.subplots(figsize=(5, 4))
name = ["With_disease", "Without_disease"]
ax = df_categoric.target.value_counts().plot(kind='bar')
ax.set_title("Heart disease", fontsize=13, weight='bold')
ax.set_xticklabels(name, rotation=0)

graph_percentage(ax)

fig, ax = plt.subplots(figsize=(8, 5))
name = ["Male", "Female"]
ax = sns.countplot(x='sex', hue='target', data=df_categoric, palette='Set2')
ax.set_title("Gender distribution according to target", fontsize=13, weight='bold')
ax.set_xticklabels(name, rotation=0)

graph_percentage(ax)

fig, ax = plt.subplots(figsize=(10, 5))
name = ["Non-anginous_pain", "Atypic_Angina","Typical_Angina ", "Asymptomatic"]
ax = sns.countplot(x='cp', hue='target', data=df_categoric, palette='Set2')
ax.set_title("Chest pain distribution according to target", fontsize=13, weight='bold')
ax.set_xticklabels(name, rotation=0)

graph_percentage(ax)

sns.countplot(x='fbs', hue='target', data=df_categoric, palette='Set2').set_title(
    'Blood glucose distribution according to target')
plt.show()

df_categoric2 = df.copy()
df_categoric2['fbs'] = df_categoric2.fbs.replace({0: "True", 1: "False"})

fig, ax = plt.subplots(figsize=(10, 5))
name = ["False", "True"]
ax = sns.countplot(x='fbs', hue='target', data=df_categoric2, palette='Set2')
ax.set_title("Blood glucose distribution > 120 mg/dl according to the target", fontsize=13, weight='bold')
ax.set_xticklabels(name, rotation=0)

graph_percentage(ax)

fig, ax = plt.subplots(figsize=(10, 5))
name = ["Increasing", "Decreasing", "Flat"]
ax = sns.countplot(x='slope', hue='target', data=df_categoric, palette='Set2')
ax.set_title("Slope distribution according to the target", fontsize=13, weight='bold')
ax.set_xticklabels(name, rotation=0)

graph_percentage(ax)

fig, ax = plt.subplots(figsize=(10, 5))
name = ["Yes", "No"]
ax = sns.countplot(x='exang', hue='target', data=df_categoric, palette='Set2')
ax.set_title("Distribution of exercise-induced angina (exang) according to target", fontsize=13, weight='bold')
ax.set_xticklabels(name, rotation=0)

graph_percentage(ax)

fig, ax = plt.subplots(figsize=(10, 5))
name = ["Fixed_defect", "Reversible_defect", "Normal"]
ax = sns.countplot(x='thal', hue='target', data=df_categoric, palette='Set2')
ax.set_title("Thal distribution according to the target", fontsize=13, weight='bold')
ax.set_xticklabels(name, rotation=0)

graph_percentage(ax)


print("\nCORRELATION MATRIX WITH OUTLIERS\n")
for i, colum in enumerate(['pearson', 'spearman', 'kendall']):
    df_corr = df.corr(method=colum)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    sns.heatmap(
        df_corr,
        annot=True,
        cbar=False,
        annot_kws={"size": 8},
        vmin=-1,
        vmax=1,
        center=0,
        cmap=sns.diverging_palette(350, 350, n=200),
        square=True,
        ax=ax
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    ax.tick_params(labelsize=10)
    plt.show()

print("\nCORRELATION MATRIX WITHOUT OUTLIERS\n")
for i, colum in enumerate(['pearson', 'spearman', 'kendall']):
    df_corr = df_without_outlier.corr(method=colum)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    sns.heatmap(
        df_corr,
        annot=True,
        cbar=False,
        annot_kws={"size": 8},
        vmin=-1,
        vmax=1,
        center=0,
        cmap=sns.diverging_palette(350, 350, n=200),
        square=True,
        ax=ax
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    ax.tick_params(labelsize=10)
    plt.show()



classifier = LogisticRegression(random_state=0)
print("\nLinear logistic regression for normal dataset:")
logistic(df, classifier)

print("\nLinear logistic regression with non-categorical data from a normal dataset:")
no_categoric_logistic(df, classifier)

print("\nLinear logistic regression for dataset without outliers:")
logistic(df_without_outlier, classifier)

print("\nLinear logistic regression with non-categorical data from a dataset without outliers:")
no_categoric_logistic(df_without_outlier, classifier)

X = df[["cp", "thalach", "slope", "oldpeak", "exang", "ca", "thal", "sex", "age"]].values
y = df["target"].values

print("\nBest K for normal dataset:")
best_k_df_normal = best_k(X, y)

X = df[["age", "oldpeak"]].values
y = df["target"].values

print("\nBest K for dataset with normal non-categorical data:")
best_k_df_normal_no_categoric = best_k(X, y)

X = df_without_outlier[["cp", "thalach", "slope", "oldpeak", "exang", "ca", "thal", "sex", "age"]].values
y = df_without_outlier["target"].values

print("\nBest K for dataset without outliers:")
best_k_df_without_outliers = best_k(X, y)

X = df_without_outlier[["age", "oldpeak"]].values
y = df_without_outlier["target"].values

print("\nBest K for dataset with non-categorical data without outliers:")
best_k_df_without_outliers_no_categoric = best_k(X, y)

classifier_df_normal = KNeighborsClassifier(n_neighbors=best_k_df_normal, metric="minkowski", p=2)
classifier_df_normal_no_categoric = KNeighborsClassifier(n_neighbors=best_k_df_normal_no_categoric, metric="minkowski", p=2)
classifier_df_sin_outlier = KNeighborsClassifier(n_neighbors=best_k_df_without_outliers, metric="minkowski", p=2)
classifier_df_sin_outlier_no_categoric = KNeighborsClassifier(n_neighbors=best_k_df_without_outliers_no_categoric, metric="minkowski", p=2)

print("\nKNN logistic regression for normal dataset:")
logistic(df, classifier_df_normal)

print("\nKNN logistics for dataset with non-categorical data normal:")
no_categoric_logistic(df, classifier_df_normal_no_categoric)

print("\nKNN logistic regression for dataset without outlier:")
logistic(df_without_outlier, classifier_df_sin_outlier)

print("\nKNN logistics for dataset with non-categorical data without outlier:")
no_categoric_logistic(df_without_outlier, classifier_df_sin_outlier_no_categoric)


classifier_svc_linear = SVC(kernel="linear", random_state=0)
classifier_svc_poly = SVC(kernel="poly", random_state=0)
classifier_svc_rbf = SVC(kernel="rbf", random_state=0)
classifier_svc_sigmoid = SVC(kernel="sigmoid", random_state=0)

print("\nClassifier_svc_linear logistic regression for normal dataset:")
logistic(df, classifier_svc_linear)

print("\nClassifier_svc_poly logistic regression for normal dataset:")
logistic(df, classifier_svc_poly)

print("\nLogistic regression classifier_svc_rbf for normal dataset:")
logistic(df, classifier_svc_rbf)

print("\nLogistic regression classifier_svc_sigmoid for normal dataset:")
logistic(df, classifier_svc_sigmoid)

print("\nLogistic regression classifier_svc_linear for dataset with non-categorical data normal:")
no_categoric_logistic(df, classifier_svc_linear)

print("\nLogistic regression classifier_svc_poly for dataset with normal non-categorical data:")
no_categoric_logistic(df, classifier_svc_poly)

print("\nLogistic regression classifier_svc_rbf for dataset with non-categorical data normal:")
no_categoric_logistic(df, classifier_svc_rbf)

print("\nLogistic regression classifier_svc_sigmoid for dataset with non-categorical data normal:")
no_categoric_logistic(df, classifier_svc_sigmoid)

print("\nLogistic regression classifier_svc_linear for dataset without outlier:")
logistic(df_without_outlier, classifier_svc_linear)

print("\nLogistic regression classifier_svc_poly for dataset without outlier:")
logistic(df_without_outlier, classifier_svc_poly)

print("\nLogistic regression classifier_svc_rbf for dataset without outlier:")
logistic(df_without_outlier, classifier_svc_rbf)

print("\nLogistic regression classifier_svc_sigmoid for dataset without outlier:")
logistic(df_without_outlier, classifier_svc_sigmoid)

print("\nLogistic regression classifier_svc_linear for dataset with non-categorical data without outlier:")
no_categoric_logistic(df_without_outlier, classifier_svc_linear)

print("\nLogistic regression classifier_svc_poly for dataset with non-categorical data without outlier:")
no_categoric_logistic(df_without_outlier, classifier_svc_poly)


print("\nLogistic regression classifier_svc_rbf for dataset with non-categorical data without outlier:")
no_categoric_logistic(df_without_outlier, classifier_svc_rbf)

print("\nLogistic regression classifier_svc_sigmoid for dataset with non-categorical data without outlier:")
no_categoric_logistic(df_without_outlier, classifier_svc_sigmoid)

classifier = GaussianNB()

print("\nNaive Bayes logistic regression for normal dataset:")
logistic(df, classifier)

print("\nNaive Bayes logistic regression for dataset with non-categorical normal data:")
no_categoric_logistic(df, classifier)

print("\nNaive Bayes logistic regression for dataset without outlier:")
logistic(df_without_outlier, classifier)

print("\nNaive Bayes logistic regression for dataset with non-categorical data without outlier:")
no_categoric_logistic(df_without_outlier, classifier)


classifier_tree = DecisionTreeClassifier(criterion="entropy", random_state=0)

print("\nLogistic regression with decision trees for normal dataset:")
logistic(df, classifier_tree)

print("\nLogistic regression with decision trees for dataset with non-categorical normal data:")
no_categoric_logistic(df, classifier_tree)

print("\nLogistic regression with decision trees for dataset without outlier:")
logistic(df_without_outlier, classifier_tree)

print("\nLogistic regression with decision trees for dataset with non-categorical data without outlier:")
no_categoric_logistic(df_without_outlier, classifier_tree)


print("\nRandom forest for dataset with outliers:")
max_depth_random_forest(df)

print("\nRandom forest for dataset without outliers:")
max_depth_random_forest(df_without_outlier)

print("\nRandom forest for dataset with non-categorical data with outliers:")
no_categoric_max_depth_random_forest(df)

print("\nRandom forest graph for dataset with non-categorical data without outliers:")
no_categoric_max_depth_random_forest(df_without_outlier)