import numpy as np
import data_process
from model import LeNet5
from model import softmax_loss
from optimizer import Adam
import tqdm
import matplotlib.pyplot as plt

batch_size = 200
epochs = 20
learning_rate = 1e-3

mnist_dir = "mnist_data/"
train_data_dir = "train-images-idx3-ubyte"
train_label_dir = "train-labels-idx1-ubyte"
test_data_dir = "t10k-images-idx3-ubyte"
test_label_dir = "t10k-labels-idx1-ubyte"
data = data_process.load_data(mnist_dir, train_data_dir, train_label_dir, test_data_dir, test_label_dir)
model = LeNet5()
optimizer = Adam(model.get_params(), learning_rate)

history_loss = []
history_acc = []

def train_model():
    for k, v in list(data.items()):
        print(f"{k}: {v.shape}")
    best_acc = 0
    best_weight = None
    for e in range(epochs):
        # add tqdm
        pbar = tqdm.tqdm(range(0, int(data['X_train'].shape[0]/batch_size)), ncols=150)
        for i in pbar:
            X, y = data_process.get_batch(data["X_train"], data["y_train"], batch_size)
            y_pred = model.forward(X)
            loss, grad, acc = softmax_loss(y_pred, y)
            history_loss.append(loss)
            history_acc.append(acc)
            model.backward(grad)
            optimizer.step()
            pbar.set_description(f"Epoch: {e+1}/{epochs}")
            pbar.set_postfix(loss=loss, acc=acc)

        val_X = data["X_val"]
        val_y = data["y_val"]
        y_pred = model.forward(val_X)
        y_pred = np.argmax(y_pred, axis=1)
        acc = np.mean(y_pred == val_y.reshape(1, val_y.shape[0]))
        if acc > best_acc:
            best_acc = acc
            best_weight = model.get_params()
        pbar.set_postfix(val_acc=acc)
    return best_weight


def test(best_weight):
    X = data["X_test"]
    y = data["y_test"]
    model.set_params(best_weight)
    y_pred = model.forward(X)
    y_pred = np.argmax(y_pred, axis=1)
    acc = np.mean(y_pred == y.reshape(1, y.shape[0]))
    print(f"Test Accuracy: {acc*100}%")


def plot_result():
    # 分别绘制 loss 和 accuracy 曲线 并保存图片
    plt.figure()
    plt.plot(history_loss)
    plt.title("history loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig('loss.png')
    plt.figure()
    plt.plot(history_acc)
    plt.title("history acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.savefig('acc.png')