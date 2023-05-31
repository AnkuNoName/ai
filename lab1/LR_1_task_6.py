import numpy as np
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

# Завантаження даних з файлу
data = np.loadtxt('data_multivar_nb.txt', delimiter=',')

# Розділення вхідних ознак та міток класів
X = data[:, :-1]
y = data[:, -1]

# Розділення даних на тренувальний та тестувальний набори
# Ви можете використовувати свою власну стратегію розділення, наприклад, train_test_split з scikit-learn
X_train = X[:80]
y_train = y[:80]
X_test = X[80:]
y_test = y[80:]

# Навчання SVM
svm_model = svm.SVC()
svm_model.fit(X_train, y_train)

# Навчання наївного байєсівського класифікатора
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Класифікація за допомогою SVM
svm_predictions = svm_model.predict(X_test)

# Класифікація за допомогою наївного байєсівського класифікатора
nb_predictions = nb_model.predict(X_test)

# Оцінка якості класифікації для SVM
svm_report = classification_report(y_test, svm_predictions)
svm_confusion_matrix = confusion_matrix(y_test, svm_predictions)

# Оцінка якості класифікації для наївного байєсівського класифікатора
nb_report = classification_report(y_test, nb_predictions)
nb_confusion_matrix = confusion_matrix(y_test, nb_predictions)

# Виведення результатів
print("SVM Classification Report:")
print(svm_report)
print("SVM Confusion Matrix:")
print(svm_confusion_matrix)

print("Naive Bayes Classification Report:")
print(nb_report)
print("Naive Bayes Confusion Matrix:")
print(nb_confusion_matrix)