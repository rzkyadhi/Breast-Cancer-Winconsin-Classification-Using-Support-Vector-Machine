import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer_dataset = datasets.load_breast_cancer()

x = cancer_dataset.data
y = cancer_dataset.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

#SVM Classifier
classifier_svm = svm.SVC(kernel="linear")
classifier_svm.fit(x_train, y_train)
y_pred = classifier_svm.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)

#k-NN Classifier
classifier_knn = KNeighborsClassifier(n_neighbors=5)
classifier_knn.fit(x_train, y_train)
y_pred_knn = classifier_knn.predict(x_test)

accuracy_knn = metrics.accuracy_score(y_test, y_pred_knn)
print(accuracy_knn)