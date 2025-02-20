import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, \
    classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def drop_na(data):
    data = data.dropna(axis=0)

    print("\nNUMBER OF NULL VALUES PER COLUMN")
    print(data.isnull().sum())

    sns.heatmap(data.isnull(), cbar=False)
    plt.show()

    return data


def outliers(df, ft):
    q1 = df[ft].quantile(0.25)
    q3 = df[ft].quantile(0.75)
    iqr = q3 - q1

    low = q1 - 1.5 * iqr
    up = q3 + 1.5 * iqr

    ls = df.index[(df[ft] < low) | (df[ft] > up)]

    return ls


def remove(df, ls):
    ls = sorted(set(ls))
    df = df.drop(ls)
    return df


def delete_outliers(n, df, list):
    df_clean = df
    for i in range(n):
        index_list = []
        for feature in list:
            index_list.extend(outliers(df_clean, feature))
        if not index_list:
            break
        df_clean = remove(df_clean, index_list)
    return df_clean


def graph_percentage(ax):
    totals = []
    for i in ax.patches:
        totals.append(i.get_height())
    total = sum(totals)
    for i in ax.patches:
        ax.text(i.get_x(), i.get_height() - 5,
                str(round((i.get_height() / total) * 100, 2)) + '%', fontsize=14,
                color='white', weight='bold')

    plt.tight_layout()
    plt.show()


def logistic(df, classifier):
    X = df[["cp", "thalach", "slope", "oldpeak", "exang", "ca", "thal", "sex", "age"]].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm,
                annot=True,
                fmt='g',
                cmap='Purples')
    plt.ylabel('Prediction', fontsize=13)
    plt.xlabel('Current', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17)
    plt.show()

    print(f'\nModel precision: {precision_score(y_test, y_pred):.2f}')
    print(f'Model accuracy:{accuracy_score(y_test, y_pred):.2f}')
    print(f'Model sensitivity: {recall_score(y_test, y_pred):.2f}')
    print(f'Model F1 score:{f1_score(y_test, y_pred):.2f}')
    print(f'ROC - AUC curve of the model:{roc_auc_score(y_test, y_pred):.2f}\n')

    print(classification_report(y_test, y_pred))


def no_categoric_logistic(df, classifier):
    X = df[["age", "oldpeak"]].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm,
                annot=True,
                fmt='g',
                cmap='Purples')
    plt.ylabel('Prediction', fontsize=13)
    plt.xlabel('Current', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17)
    plt.show()

    print(f'\nModel precision: {precision_score(y_test, y_pred):.2f}')
    print(f'Model accuracy:{accuracy_score(y_test, y_pred):.2f}')
    print(f'Model sensitivity: {recall_score(y_test, y_pred):.2f}')
    print(f'Model F1 score:{f1_score(y_test, y_pred):.2f}')
    print(f'ROC - AUC curve of the model:{roc_auc_score(y_test, y_pred):.2f}\n')

    print(classification_report(y_test, y_pred))

    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    color=ListedColormap(('red', 'green'))(i), label=j)
    plt.title('Classifier (Training Set)')
    plt.xlabel('Age')
    plt.ylabel('Oldpeak')
    plt.legend()
    plt.show()

    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    color=ListedColormap(('red', 'green'))(i), label=j)
    plt.title('Classifier (Test Set)')
    plt.xlabel('Age')
    plt.ylabel('Oldpeak')
    plt.legend()
    plt.show()


def best_k(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    media_pre = np.zeros((9))
    for n in range(1, 10):
        classifier = KNeighborsClassifier(n_neighbors=n, metric="minkowski", p=2)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        media_pre[n - 1] = accuracy_score(y_test, y_pred)

    print(f"Better precision: {media_pre.max()}, with K = {media_pre.argmax() + 1}")

    return media_pre.argmax() + 1


def max_depth_random_forest(df):
    X = df[["cp", "thalach", "slope", "oldpeak", "exang", "ca", "thal", "sex", "age"]].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # Max_depth values ​​to evaluate.
    max_depth_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    # List to store the cross-validation results.
    scores = []

    # Iterate over the max_depth values.
    for max_depth in max_depth_values:
        rf_classifier = RandomForestClassifier(max_depth=max_depth)

        cv_scores = cross_val_score(rf_classifier, X, y, cv=5)

        scores.append(np.mean(cv_scores))

    # Find the index of the best value of max_depth.
    best_index = np.argmax(scores)
    best_max_depth = max_depth_values[best_index]

    print("Cross-validation results:")
    print("Best max_depth: {}".format(best_max_depth))

    # Once the best depth has been found, we proceed to perform the classification.

    classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0, max_depth=best_max_depth)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm,
                annot=True,
                fmt='g',
                cmap='Purples')
    plt.ylabel('Prediction', fontsize=13)
    plt.xlabel('Current', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17)
    plt.show()

    print(f'\nModel precision: {precision_score(y_test, y_pred):.2f}')
    print(f'Model accuracy:{accuracy_score(y_test, y_pred):.2f}')
    print(f'Model sensitivity: {recall_score(y_test, y_pred):.2f}')
    print(f'Model F1 score:{f1_score(y_test, y_pred):.2f}')
    print(f'ROC - AUC curve of the model:{roc_auc_score(y_test, y_pred):.2f}\n')

    print(classification_report(y_test, y_pred))


def no_categoric_max_depth_random_forest(df):
    X = df[["age", "oldpeak"]].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # Max_depth values ​​to evaluate.
    max_depth_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    # List to store the cross-validation results.
    scores = []

    # Iterate over the max_depth values.
    for max_depth in max_depth_values:
        rf_classifier = RandomForestClassifier(max_depth=max_depth)

        cv_scores = cross_val_score(rf_classifier, X, y, cv=5)

        scores.append(np.mean(cv_scores))

    # Find the index of the best value of max_depth.
    best_index = np.argmax(scores)
    best_max_depth = max_depth_values[best_index]

    print("Resultados de la validación cruzada:")
    print("Mejor max_depth: {}".format(best_max_depth))

    # Once the best depth has been found, we proceed to perform the classification.

    classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0, max_depth=best_max_depth)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm,
                annot=True,
                fmt='g',
                cmap='Purples')
    plt.ylabel('Prediction', fontsize=13)
    plt.xlabel('Current', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17)
    plt.show()

    print(f'\nModel precision: {precision_score(y_test, y_pred):.2f}')
    print(f'Model accuracy:{accuracy_score(y_test, y_pred):.2f}')
    print(f'Model sensitivity: {recall_score(y_test, y_pred):.2f}')
    print(f'Model F1 score:{f1_score(y_test, y_pred):.2f}')
    print(f'ROC - AUC curve of the model:{roc_auc_score(y_test, y_pred):.2f}\n')

    print(classification_report(y_test, y_pred))

    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    color=ListedColormap(('red', 'green'))(i), label=j)
    plt.title('Classifier (Training Set)')
    plt.xlabel('Age')
    plt.ylabel('Oldpeak')
    plt.legend()
    plt.show()

    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    color=ListedColormap(('red', 'green'))(i), label=j)
    plt.title('Classifier (Test Set)')
    plt.xlabel('Age')
    plt.ylabel('Oldpeak')
    plt.legend()
    plt.show()