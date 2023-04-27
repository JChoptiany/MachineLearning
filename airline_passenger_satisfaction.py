import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    # Problem: airline passenger satisfaction classification based on:
    #           - passenger data (gender, age, loyal customer status)
    #           - flight details (distane, class, delay) 
    #           - passenger's rating (on e.g. seat comfort, service quality, Wi-Fi access)

    # data source: https://www.kaggle.com/datasets/teejmahal20/

    # load and prepare dataset
    df = pd.read_csv('data\\train.csv')
    df.drop(columns=['Unnamed: 0', 'id'], inplace=True)
    df.dropna(inplace=True)

    target_encoding = {'neutral or dissatisfied': 0, 'satisfied': 1}
    df['satisfaction'] = df['satisfaction'].map(target_encoding).astype('int64')

    target_encoding = {'Female': 0, 'Male': 1}
    df['Gender'] = df['Gender'].map(target_encoding).astype('int64')

    target_encoding = {'Personal Travel': 0, 'Business travel': 1}
    df['Type of Travel'] = df['Type of Travel'].map(target_encoding).astype('int64')

    target_encoding = {'disloyal Customer': 0, 'Loyal Customer': 1}
    df['Customer Type'] = df['Customer Type'].map(target_encoding).astype('int64')

    target_encoding = {'Eco': 0, 'Eco Plus': 1, 'Business': 2}
    df['Class'] = df['Class'].map(target_encoding).astype('int64')

    y = df['satisfaction']
    x = df.drop(columns=['satisfaction'], inplace=False)

    # split dataset to train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # model
    random_forest = RandomForestClassifier()
    random_forest.fit(x_train, y_train)

    # prediction
    y_prediction = random_forest.predict(x_test)
    predict_proba = random_forest.predict_proba(x_test)

    # scores
    print('Classification scores:')
    print(f'\tAccuracy: {accuracy_score(y_test, y_prediction)}')
    print(f'\tF-1: {f1_score(y_test, y_prediction)}')
    print(f'\tPrecision: {precision_score(y_test, y_prediction)}')
    fpr, tpr, thresholds = roc_curve(y_test, predict_proba[:, 1])
    roc_auc = roc_auc_score(y_test, predict_proba[:, 1])
    print(f'\t')

    # ROC Curve and AUC
    plt.plot(fpr, tpr)
    plt.title(f'ROC Curve (Area under the ROC curve: {roc_auc})')
    plt.show()

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_prediction)
    sns.heatmap(data=conf_matrix, annot=True)
    plt.show()
