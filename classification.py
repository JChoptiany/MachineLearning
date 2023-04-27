import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, roc_auc_score


def print_scores(title: str, y_true, y_predict, probabilities) -> None:
    print(f'{title}:')
    print(f'\tAccuracy: {accuracy_score(y_true, y_predict)}')
    print(f'\tF-1: {f1_score(y_true, y_predict, average="macro")}')
    print(f'\tPrecision: {precision_score(y_true, y_predict, average="macro")}')
    if probabilities.shape[1] == 2:
        probabilities = probabilities[:,1]
    roc_auc = roc_auc_score(y_true, probabilities, multi_class='ovo', average='macro')
    print(f'\tArea under the ROC curve: {roc_auc}')    


if __name__ == '__main__':
    # load datasets
    iris = sklearn.datasets.load_iris()
    iris_x = iris.data
    iris_y = iris.target

    wine = sklearn.datasets.load_wine()
    wine_x = wine.data
    wine_y = wine.target

    cancer = sklearn.datasets.load_breast_cancer()
    cancer_x = cancer.data
    cancer_y = cancer.target


    # split data into train and test sets
    iris_x_train, iris_x_test, iris_y_train, iris_y_test = train_test_split(iris_x, iris_y)
    wine_x_train, wine_x_test, wine_y_train, wine_y_test = train_test_split(wine_x, wine_y)
    cancer_x_train, cancer_x_test, cancer_y_train, cancer_y_test = train_test_split(cancer_x, cancer_y)
    

    # train models using different models

    # K-Nearest Neighbours
    knn_iris = KNeighborsClassifier()
    knn_iris.fit(iris_x_train, iris_y_train)

    knn_wine = KNeighborsClassifier()
    knn_wine.fit(wine_x_train, wine_y_train)

    knn_cancer = KNeighborsClassifier()
    knn_cancer.fit(cancer_x_train, cancer_y_train)

    # Support Vector Machines
    svm_iris = SVC(probability=True)
    svm_iris.fit(iris_x_train, iris_y_train)
    
    svm_wine = SVC(probability=True)
    svm_wine.fit(wine_x_train, wine_y_train)

    svm_cancer = SVC(probability=True)
    svm_cancer.fit(cancer_x_train, cancer_y_train)

    # Decision Tree
    tree_iris = DecisionTreeClassifier()
    tree_iris.fit(iris_x_train, iris_y_train)

    tree_wine = DecisionTreeClassifier()
    tree_wine.fit(wine_x_train, wine_y_train)

    tree_cancer = DecisionTreeClassifier()
    tree_cancer.fit(cancer_x_train, cancer_y_train)

    # Gaussian Naive Bayes
    bayes_iris = GaussianNB()
    bayes_iris.fit(iris_x_train, iris_y_train)

    bayes_wine = GaussianNB()
    bayes_wine.fit(wine_x_train, wine_y_train)

    bayes_cancer = GaussianNB()
    bayes_cancer.fit(cancer_x_train, cancer_y_train)


    # prediction
    knn_iris_prediction = knn_iris.predict(iris_x_test)
    knn_wine_prediction = knn_wine.predict(wine_x_test)
    knn_cancer_prediction = knn_cancer.predict(cancer_x_test)

    svm_iris_prediction = svm_iris.predict(iris_x_test)
    svm_wine_prediction = svm_wine.predict(wine_x_test)
    svm_cancer_prediction = svm_cancer.predict(cancer_x_test)

    tree_iris_prediction = tree_iris.predict(iris_x_test)
    tree_wine_prediction = tree_wine.predict(wine_x_test)
    tree_cancer_prediction = tree_cancer.predict(cancer_x_test)

    bayes_iris_prediction = bayes_iris.predict(iris_x_test)
    bayes_wine_prediction = bayes_wine.predict(wine_x_test)
    bayes_cancer_prediction = bayes_cancer.predict(cancer_x_test)


    # scores        
    knn_iris_probabilities = knn_iris.predict_proba(iris_x_test) 
    knn_wine_probabilities = knn_wine.predict_proba(wine_x_test) 
    knn_cancer_probabilities = knn_cancer.predict_proba(cancer_x_test) 
    svm_iris_probabilities = svm_iris.predict_proba(iris_x_test) 
    svm_wine_probabilities = svm_wine.predict_proba(wine_x_test) 
    svm_cancer_probabilities = svm_cancer.predict_proba(cancer_x_test) 
    tree_iris_probabilities = tree_iris.predict_proba(iris_x_test) 
    tree_wine_probabilities = tree_wine.predict_proba(wine_x_test) 
    tree_cancer_probabilities = tree_cancer.predict_proba(cancer_x_test) 
    bayes_iris_probabilities = bayes_iris.predict_proba(iris_x_test) 
    bayes_wine_probabilities = bayes_wine.predict_proba(wine_x_test) 
    bayes_cancer_probabilities = bayes_cancer.predict_proba(cancer_x_test) 

    print('=========== Scores ===========')
    print_scores('KNN Iris', iris_y_test, knn_iris_prediction, knn_iris_probabilities)
    print_scores('KNN Wine', wine_y_test, knn_wine_prediction, knn_wine_probabilities)
    print_scores('KNN Breast cancer', cancer_y_test, knn_cancer_prediction, knn_cancer_probabilities)
    print()
    print_scores('SVM Iris', iris_y_test, svm_iris_prediction, svm_iris_probabilities)
    print_scores('SVM Wine', wine_y_test, svm_wine_prediction, svm_wine_probabilities)
    print_scores('SVM Breast cancer', cancer_y_test, svm_cancer_prediction, svm_cancer_probabilities)
    print()
    print_scores('DT Iris', iris_y_test, tree_iris_prediction, tree_iris_probabilities)
    print_scores('DT Wine', wine_y_test, tree_wine_prediction, tree_wine_probabilities)
    print_scores('DT Breast cancer', cancer_y_test, tree_cancer_prediction, tree_cancer_probabilities)
    print()
    print_scores('NB Iris', iris_y_test, bayes_iris_prediction, bayes_iris_probabilities)
    print_scores('NB Wine', wine_y_test, bayes_wine_prediction, bayes_wine_probabilities)
    print_scores('NB Breast cancer', cancer_y_test, bayes_cancer_prediction, bayes_cancer_probabilities)
    print('================================')    


    # confusion matrix
    knn_iris_confusion_matrix = confusion_matrix(iris_y_test, knn_iris_prediction)
    knn_wine_confusion_matrix = confusion_matrix(wine_y_test, knn_wine_prediction)
    knn_cancer_confusion_matrix = confusion_matrix(cancer_y_test, knn_cancer_prediction)
    
    svm_iris_confusion_matrix = confusion_matrix(iris_y_test, svm_iris_prediction)
    svm_wine_confusion_matrix = confusion_matrix(wine_y_test, svm_wine_prediction)
    svm_cancer_confusion_matrix = confusion_matrix(cancer_y_test, svm_cancer_prediction)
    
    tree_iris_confusion_matrix = confusion_matrix(iris_y_test, tree_iris_prediction)
    tree_wine_confusion_matrix = confusion_matrix(wine_y_test, tree_wine_prediction)
    tree_cancer_confusion_matrix = confusion_matrix(cancer_y_test, tree_cancer_prediction)
    
    bayes_iris_confusion_matrix = confusion_matrix(iris_y_test, bayes_iris_prediction)
    bayes_wine_confusion_matrix = confusion_matrix(wine_y_test, bayes_wine_prediction)
    bayes_cancer_confusion_matrix = confusion_matrix(cancer_y_test, bayes_cancer_prediction)
    
    fig, axes = plt.subplots(3, 4)
    fig.suptitle('Confusion matrices for 3 datasets and 4 classification alghorithms')
    
    axes[2][0].set_title('KNN & Breast cancer')
    sns.heatmap(ax=axes[0][0], data=knn_iris_confusion_matrix, annot=True)
    axes[1][0].set_title('KNN & Wine')
    sns.heatmap(ax=axes[1][0], data=knn_wine_confusion_matrix, annot=True)
    axes[0][0].set_title('KNN & Iris')
    sns.heatmap(ax=axes[2][0], data=knn_cancer_confusion_matrix, annot=True)
    
    axes[2][1].set_title('SVM & Breast cancer')
    sns.heatmap(ax=axes[0][1], data=svm_iris_confusion_matrix, annot=True)
    axes[1][1].set_title('SMV & Wine')
    sns.heatmap(ax=axes[1][1], data=svm_wine_confusion_matrix, annot=True)
    axes[0][1].set_title('SMV & Iris')
    sns.heatmap(ax=axes[2][1], data=svm_cancer_confusion_matrix, annot=True)
    
    axes[2][2].set_title('DT & Breast cancer')
    sns.heatmap(ax=axes[0][2], data=tree_iris_confusion_matrix, annot=True)
    axes[1][2].set_title('DT & Wine')
    sns.heatmap(ax=axes[1][2], data=tree_wine_confusion_matrix, annot=True)
    axes[0][2].set_title('DT & Iris')
    sns.heatmap(ax=axes[2][2], data=tree_cancer_confusion_matrix, annot=True)
    
    axes[2][3].set_title('NB & Breast cancer')
    sns.heatmap(ax=axes[0][3], data=bayes_iris_confusion_matrix, annot=True)
    axes[1][3].set_title('NB & Wine')
    sns.heatmap(ax=axes[1][3], data=bayes_wine_confusion_matrix, annot=True)
    axes[0][3].set_title('NB & Iris')
    sns.heatmap(ax=axes[2][3], data=bayes_cancer_confusion_matrix, annot=True)
    
    plt.show()