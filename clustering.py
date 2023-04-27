import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.cluster import KMeans, MeanShift, OPTICS
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, calinski_harabasz_score, homogeneity_score, mutual_info_score, rand_score


def print_scores(x, y_predict, y_true):
    print(f'\tSilhouette: {silhouette_score(x, y_predict)}')
    print(f'\tCaliński-Harabasz: {calinski_harabasz_score(x, y_predict)}')
    print(f'\tRand: {rand_score(y_true, y_predict)}')
    print(f'\tHomogenity: {homogeneity_score(y_true, y_predict)}')
    print(f'\tMutual information: {mutual_info_score(y_true, y_predict)}')


if __name__ == '__main__':
    # Load datasets
    first_x, first_y = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)
    second_x, second_y = make_blobs(n_samples=300, n_features=2)


    # Show datasets
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)
    axes[0].scatter(first_x[:,0], first_x[:,1], c=first_y)
    axes[0].set_title('First dataset (make_classification)')
    axes[1].scatter(second_x[:,0], second_x[:,1], c=second_y)
    axes[1].set_title('Second dataset (make_blobs)')
    plt.show()


    # Split datasets to train and test
    first_x_train, first_x_test, first_y_train, first_y_test = train_test_split(first_x, first_y)
    second_x_train, second_x_test, second_y_train, second_y_test = train_test_split(second_x, second_y)


    # Clustering
    # K-Means
    kmeans_first = KMeans(n_clusters=2, n_init='auto')
    kmeans_first.fit(first_x_train)
    kmeans_second = KMeans(n_clusters=3, n_init='auto')
    kmeans_second.fit(second_x_train)

    kmeans_first_predict = kmeans_first.predict(first_x_test)
    kmeans_second_predict = kmeans_second.predict(second_x_test)

    # Mean Shift
    meanshift_first = MeanShift()
    meanshift_first.fit(first_x_train)
  
    meanshift_second = MeanShift()
    meanshift_second.fit(second_x_train)

    meanshift_first_predict = meanshift_first.predict(first_x_test)
    meanshift_second_predict = meanshift_second.predict(second_x_test)

    # OPTICS
    optics_first_predict = OPTICS(min_cluster_size=first_x_test.shape[0]//2).fit_predict(first_x_test)
    optics_second_predict = OPTICS(min_cluster_size=second_x_test.shape[0]//3).fit_predict(second_x_test)


    # Scores
    print('========= Scores =========')
    print('K-Mean first dataset')
    print_scores(first_x_test, kmeans_first_predict, first_y_test)
    print('K-Mean second dataset')
    print_scores(second_x_test, kmeans_second_predict, second_y_test)
    print()
    print('Mean shift first dataset')
    print_scores(first_x_test, meanshift_first_predict, first_y_test)
    print('Mean shift second dataset')
    print_scores(second_x_test, meanshift_second_predict, second_y_test)
    print()
    print('OPTICS first dataset')
    print_scores(first_x_test, optics_first_predict, first_y_test)
    print('OPTICS second dataset')
    print_scores(second_x_test, optics_second_predict, second_y_test)
    print()
    

    # Show predictions
    fig, axes = plt.subplots(4, 2)
    fig.set_size_inches(10, 5)

    axes[0][0].scatter(first_x_test[:,0], first_x_test[:,1], c=first_y_test)
    axes[0][0].set_title('True first dataset')
    axes[0][1].scatter(second_x_test[:,0], second_x_test[:,1], c=second_y_test)
    axes[0][1].set_title('True second dataset')
    
    axes[1][0].scatter(first_x_test[:,0], first_x_test[:,1], c=kmeans_first_predict)
    axes[1][0].set_title('KMeans first dataset')
    axes[1][1].scatter(second_x_test[:,0], second_x_test[:,1], c=kmeans_second_predict)
    axes[1][1].set_title('KMeans second dataset')

    axes[2][0].scatter(first_x_test[:,0], first_x_test[:,1], c=meanshift_first_predict)
    axes[2][0].set_title('Mean shift first dataset')
    axes[2][1].scatter(second_x_test[:,0], second_x_test[:,1], c=meanshift_second_predict)
    axes[2][1].set_title('Mean shift second dataset')

    axes[3][0].scatter(first_x_test[:,0], first_x_test[:,1], c=optics_first_predict)
    axes[3][0].set_title('OPTICS first dataset')
    axes[3][1].scatter(second_x_test[:,0], second_x_test[:,1], c=optics_second_predict)
    axes[3][1].set_title('OPTICS second dataset')
    plt.show()
