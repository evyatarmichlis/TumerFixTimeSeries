import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import ShrunkCovariance
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

class OHIT:
    def __init__(self, shrinkage=0.1, n_neighbors=5):
        self.shrinkage = shrinkage
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.minority_class = np.argmin(np.bincount(y))
        self.majority_class = np.argmax(np.bincount(y))

        X_minority = X[y == self.minority_class]
        X_majority = X[y == self.majority_class]

        self.cluster_modes(X_minority)
        synthetic_samples = self.generate_synthetic_samples(X_minority)

        self.X_resampled = np.vstack((X, synthetic_samples))
        self.y_resampled = np.hstack((y, np.full(len(synthetic_samples), self.minority_class)))

        return self

    def cluster_modes(self, X_minority):
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X_minority)
        self.modes = nbrs.kneighbors_graph(X_minority).toarray().astype(bool)

    def generate_synthetic_samples(self, X_minority):
        synthetic_samples = []
        for mode in self.modes:
            samples_in_mode = X_minority[mode]
            scm = ShrunkCovariance(shrinkage=self.shrinkage).fit(samples_in_mode)
            mean = np.mean(samples_in_mode, axis=0)
            synthetic_sample = np.random.multivariate_normal(mean, scm.covariance_, size=len(samples_in_mode))
            synthetic_samples.append(synthetic_sample)
        return np.vstack(synthetic_samples)


def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
x_test, y_test = readucr(root_url + "FordA_TEST.tsv")
classes = np.unique(np.concatenate((y_train, y_test), axis=0))

plt.figure()
for c in classes:
    c_x_train = x_train[y_train == c]
    plt.plot(c_x_train[0], label="class " + str(c))
plt.legend(loc="best")
plt.show()
plt.close()

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

num_classes = len(np.unique(y_train))
print(num_classes)
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

y_train[y_train == -1] = 0
y_test[y_test == -1] = 0


def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)
    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)
    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def create_imbalanced_dataset(X, y, minority_percentage):
    minority_class = np.argmin(np.bincount(y))
    majority_class = np.argmax(np.bincount(y))

    X_minority = X[y == minority_class]
    y_minority = y[y == minority_class]
    X_majority = X[y == majority_class]
    y_majority = y[y == majority_class]

    n_minority = int(len(X_minority) * minority_percentage)
    X_minority_imbalanced = X_minority[:n_minority]
    y_minority_imbalanced = y_minority[:n_minority]

    X_imbalanced = np.vstack((X_majority, X_minority_imbalanced))
    y_imbalanced = np.hstack((y_majority, y_minority_imbalanced))

    return X_imbalanced, y_imbalanced


def train_and_evaluate(X_train, y_train, X_test, y_test):
    model = make_model(input_shape=X_train.shape[1:])
    keras.utils.plot_model(model, show_shapes=True)
    epochs = 500
    batch_size = 32

    callbacks = [
        keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_loss"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1,
    )

    model = keras.models.load_model("best_model.keras")
    test_loss, test_acc = model.evaluate(X_test, y_test)

    print("Test accuracy", test_acc)
    print("Test loss", test_loss)

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate weighted precision, recall, and F1-score
    precision_weighted = precision_score(y_test, y_pred_classes, average='weighted')
    recall_weighted = recall_score(y_test, y_pred_classes, average='weighted')
    f1_weighted = f1_score(y_test, y_pred_classes, average='weighted')

    # Calculate macro precision, recall, and F1-score
    precision_macro = precision_score(y_test, y_pred_classes, average='macro')
    recall_macro = recall_score(y_test, y_pred_classes, average='macro')
    f1_macro = f1_score(y_test, y_pred_classes, average='macro')

    # Calculate micro precision, recall, and F1-score
    precision_micro = precision_score(y_test, y_pred_classes, average='micro')
    recall_micro = recall_score(y_test, y_pred_classes, average='micro')
    f1_micro = f1_score(y_test, y_pred_classes, average='micro')

    print(f"Weighted Precision: {precision_weighted}")
    print(f"Weighted Recall: {recall_weighted}")
    print(f"Weighted F1-score: {f1_weighted}")

    print(f"Macro Precision: {precision_macro}")
    print(f"Macro Recall: {recall_macro}")
    print(f"Macro F1-score: {f1_macro}")

    print(f"Micro Precision: {precision_micro}")
    print(f"Micro Recall: {recall_micro}")
    print(f"Micro F1-score: {f1_micro}")

    # Print classification report
    print(classification_report(y_test, y_pred_classes))

    metric = "sparse_categorical_accuracy"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.show()
    plt.close()
# Define different levels of minority class percentages
minority_percentages = [1,0.5,0.20, 0.10, 0.01]

for percentage in minority_percentages:
    print(f"Training with {percentage*100}% minority class data")
    X_train_imbalanced, y_train_imbalanced = create_imbalanced_dataset(x_train, y_train, percentage)
    train_and_evaluate(X_train_imbalanced, y_train_imbalanced, x_test, y_test)
