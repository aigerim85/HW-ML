import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import imread
from skimage.transform import resize
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from glob import glob
from tensorflow import keras
from tensorflow.keras import layers, models



%config InlineBackend.figure_format = 'svg'

def load_notmnist(path='./notMNIST_small',letters='ABCDEFGHIJ',
                  img_shape=(28,28),test_size=0.25,one_hot=False):

    # download data if it's missing. If you have any problems, go to the urls and load it manually.
    if not os.path.exists(path):
        print("Downloading data...")
        assert os.system('curl http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz > notMNIST_small.tar.gz') == 0
        print("Extracting ...")
        assert os.system('tar -zxvf notMNIST_small.tar.gz > untar_notmnist.log') == 0

    data,labels = [],[]
    print("Parsing...")
    for img_path in glob(os.path.join(path,'*/*')):
        class_i = img_path.split(os.sep)[-2]
        if class_i not in letters:
            continue
        try:
            data.append(resize(imread(img_path), img_shape))
            labels.append(class_i,)
        except:
            print("found broken img: %s [it's ok if <10 images are broken]" % img_path)

    data = np.stack(data)[:,None].astype('float32')
    data = (data - np.mean(data)) / np.std(data)

    #convert classes to ints
    letter_to_i = {l:i for i,l in enumerate(letters)}
    labels = np.array(list(map(letter_to_i.get, labels)))

    if one_hot:
        labels = (np.arange(np.max(labels) + 1)[None,:] == labels[:, None]).astype('float32')

    #split into train/test
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, stratify=labels)

    print("Done")
    return X_train, y_train, X_test, y_test




def plot_letters(X, y_true, y_pred=None, n=4, random_state=123):
    np.random.seed(random_state)
    indices = np.random.choice(np.arange(X.shape[0]), size=n*n, replace=False)
    plt.figure(figsize=(10, 10))
    for i in range(n*n):
        plt.subplot(n, n, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[indices[i]].reshape(28, 28), cmap='gray')
        # plt.imshow(train_images[i], cmap=plt.cm.binary)
        if y_pred is None:
            title = chr(ord("A") + y_true[indices[i]])
        else:
            title = f"y={chr(ord('A') + y_true[indices[i]])}, ŷ={chr(ord('A') + y_pred[indices[i]])}"
        plt.title(title, size=20)
    plt.show()

X_train, y_train, X_test, y_test = load_notmnist(letters='ABCDEFGHIJ')
X_train, X_test = X_train.reshape([-1, 784]), X_test.reshape([-1, 784])


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def plot_random_samples(X, y_true, y_pred):
    plot_letters(X, y_true, y_pred, n=4, random_state=912)


# Logistic Regression
logreg = LogisticRegression(max_iter=500)
%time logreg.fit(X_train, y_train)
y_train_pred_logreg = logreg.predict(X_train)
y_test_pred_logreg = logreg.predict(X_test)
print(f"Logistic Regression - Train Accuracy: {accuracy_score(y_train, y_train_pred_logreg):.2f}")
print(f"Logistic Regression - Test Accuracy: {accuracy_score(y_test, y_test_pred_logreg):.2f}")

plot_confusion_matrix(y_test, y_test_pred_logreg, labels=np.unique(y_test))

plot_random_samples(X_test, y_test, y_test_pred_logreg)



# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
%time knn.fit(X_train, y_train)
y_train_pred_knn = knn.predict(X_train)
y_test_pred_knn = knn.predict(X_test)
print(f"K-Nearest Neighbors - Train Accuracy: {accuracy_score(y_train, y_train_pred_knn):.2f}")
print(f"K-Nearest Neighbors - Test Accuracy: {accuracy_score(y_test, y_test_pred_knn):.2f}")

plot_confusion_matrix(y_test, y_test_pred_knn, labels=np.unique(y_test))

plot_random_samples(X_test, y_test, y_test_pred_knn)



# Naive Bayes
nb = GaussianNB()
%time nb.fit(X_train, y_train)
y_train_pred_nb = nb.predict(X_train)
y_test_pred_nb = nb.predict(X_test)
print(f"Naive Bayes - Train Accuracy: {accuracy_score(y_train, y_train_pred_nb):.2f}")
print(f"Naive Bayes - Test Accuracy: {accuracy_score(y_test, y_test_pred_nb):.2f}")

plot_confusion_matrix(y_test, y_test_pred_nb, labels=np.unique(y_test))

plot_random_samples(X_test, y_test, y_test_pred_nb)



# Decision Tree
dt = DecisionTreeClassifier()
%time dt.fit(X_train, y_train)
y_train_pred_dt = dt.predict(X_train)
y_test_pred_dt = dt.predict(X_test)
print(f"Decision Tree - Train Accuracy: {accuracy_score(y_train, y_train_pred_dt):.2f}")
print(f"Decision Tree - Test Accuracy: {accuracy_score(y_test, y_test_pred_dt):.2f}")

plot_confusion_matrix(y_test, y_test_pred_dt, labels=np.unique(y_test))

plot_random_samples(X_test, y_test, y_test_pred_dt)



# Random Forest
rf = RandomForestClassifier(n_estimators=100)
%time rf.fit(X_train, y_train)
y_train_pred_rf = rf.predict(X_train)
y_test_pred_rf = rf.predict(X_test)
print(f"Random Forest - Train Accuracy: {accuracy_score(y_train, y_train_pred_rf):.2f}")
print(f"Random Forest - Test Accuracy: {accuracy_score(y_test, y_test_pred_rf):.2f}")

plot_confusion_matrix(y_test, y_test_pred_rf, labels=np.unique(y_test))

plot_random_samples(X_test, y_test, y_test_pred_rf)



# MLP
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
%time mlp.fit(X_train, y_train)
y_train_pred_mlp = mlp.predict(X_train)
y_test_pred_mlp = mlp.predict(X_test)
print(f"MLP - Train Accuracy: {accuracy_score(y_train, y_train_pred_mlp):.2f}")
print(f"MLP - Test Accuracy: {accuracy_score(y_test, y_test_pred_mlp):.2f}")

plot_confusion_matrix(y_test, y_test_pred_mlp, labels=np.unique(y_test))

plot_random_samples(X_test, y_test, y_test_pred_mlp)



# CNN
# Define CNN model using Keras
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 output nodes for 10 classes
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)

# Fit CNN model
%time cnn_model.fit(X_train_cnn, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Reshape input data for CNN
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)

# Evaluate CNN model
_, train_accuracy_cnn = cnn_model.evaluate(X_train_cnn, y_train, verbose=0)
_, test_accuracy_cnn = cnn_model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"CNN - Train Accuracy: {train_accuracy_cnn:.2f}")
print(f"CNN - Test Accuracy: {test_accuracy_cnn:.2f}")

y_test_pred_cnn = np.argmax(cnn_model.predict(X_test_cnn), axis=1)

# Confusion Matrix for CNN
plot_confusion_matrix(y_test, y_test_pred_cnn, labels=np.unique(y_test))

# Random Samples for CNN
plot_random_samples(X_test, y_test, y_test_pred_cnn)


model_accuracies = {
    'Logistic Regression': accuracy_score(y_test, y_test_pred_logreg),
    'K-Nearest Neighbors': accuracy_score(y_test, y_test_pred_knn),
    'Naive Bayes': accuracy_score(y_test, y_test_pred_nb),
    'Decision Tree': accuracy_score(y_test, y_test_pred_dt),
    'Random Forest': accuracy_score(y_test, y_test_pred_rf),
    'MLP': accuracy_score(y_test, y_test_pred_mlp),
    'CNN': test_accuracy_cnn
}

plt.figure(figsize=(10, 6))
plt.bar(model_accuracies.keys(), model_accuracies.values(), color=['blue', 'green', 'orange', 'red', 'purple', 'brown', 'cyan'])
plt.ylim(0.7, 1.0)
plt.ylabel('Test Accuracy')
plt.title('Model Comparison - Test Accuracy')
plt.show()
