import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm


(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=False,
    as_supervised=True,
    with_info=True,
)
print("num training samples", ds_train.cardinality().numpy())
print("num test samples", ds_test.cardinality().numpy())

train_numpy_iterator = ds_train.as_numpy_iterator()
test_numpy_iterator = ds_test.as_numpy_iterator()
ds_train.cardinality().numpy()
train_samples = 60000
test_samples = 10000
train_ds_list = []
test_ds_list = []

X_train = np.ndarray(shape=(train_samples, 28 ** 2), dtype=np.float32)
y_train = np.ndarray(shape=(train_samples, 1), dtype=np.int32)
X_test = np.ndarray(shape=(test_samples, 28 ** 2), dtype=np.float32)
y_test = np.ndarray(shape=(test_samples, 1), dtype=np.int32)


for i in tqdm.tqdm(range(train_samples)):
    sample, label = next(train_numpy_iterator)
    train_ds_list.append((sample, label))
    X_train[i, :] = np.reshape(sample, (1, 28**2))
    y_train[i,0] = label

for i in tqdm.tqdm(range(test_samples)):
    sample, label = next(test_numpy_iterator)
    test_ds_list.append((sample, label))
    X_test[i, :] = np.reshape(sample, (1, 28 ** 2))
    y_test[i, 0] = label

one_sample, one_label = train_ds_list[0]
print(f"Data shape: {one_sample.shape}", f"label: {one_label}")

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([('log_regr', LogisticRegression(verbose=1, max_iter=20000))])
pipe.fit(X=X_train, y=y_train)
pipe.score(X=X_test, y=y_test)
