import pickle
import argparse
import torch
from torch import optim
from torch.utils.data import Subset, DataLoader
import torch.nn.functional as F
from data import *
import numpy as np
import os
from model import TMC

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Mean corruption vectors 
noise_ratios = np.arange(0.6, 0.95, 0.10)

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, help='input batch size for training', default=128)
parser.add_argument('--epochs', type=int, help='number of epochs to train', default=50)
parser.add_argument('--lambda-epochs', type=int, help='gradually increase the value of lambda from 0 to 1', default=100)
parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
parser.add_argument('--dataset', type=str, help='PIE,Caltech101,Leaves', default="Caltech101")  # note
parser.add_argument('--noise_type', type=str, help='flip, sym, IDN', default="IDN")
args = parser.parse_args()


if args.dataset == "PIE":
    args.epochs = 200
    args.noise_adjust = 1.05
    args.batch_size = 128
    noise_ratios = np.arange(0.6, 0.95, 0.10)
    args.dims = [[484, 1024], [256, 512], [279, 512]]  
    dataset = PIE()
elif args.dataset == "Caltech101":
    args.epochs = 100
    noise_ratios = np.arange(0.6, 0.95, 0.1)
    args.noise_adjust = 1
    args.data_path = 'datasets/' + args.dataset
    args.dims = [[48], [40], [254],[1984],[512],[928]]  
    dataset = Caltech101()
elif args.dataset == "CUB":
    args.epochs = 100
    noise_ratios = np.arange(0.6, 0.95, 0.1)
    args.noise_adjust = 1
    args.data_path = 'datasets/' + args.dataset
    args.dims = [[1024], [300]]  
    dataset = CUB()
elif args.dataset == "UCI":
    args.epochs = 100
    noise_ratios = np.arange(0.6, 0.95, 0.10)
    args.noise_adjust = 1
    args.data_path = 'datasets/' + args.dataset
    args.dims = [[6], [240], [47]]
    dataset = UCI()
elif args.dataset == "Leaves":
    args.epochs = 200
    noise_ratios = np.arange(0.6, 0.95, 0.1)
    args.noise_adjust = 1.1
    args.data_path = 'datasets/' + args.dataset
    args.dims = [[64, 256], [64, 256], [64, 256]]
    dataset = Leaves100()
elif args.dataset == "BBC":
    args.epochs = 100
    noise_ratios = np.arange(0.6, 0.95, 0.10)
    args.noise_adjust = 0.5
    args.data_path = 'datasets/' + args.dataset
    args.dims = [[4659], [4633], [4665], [4684]]
    dataset = BBC()

def ModelTest(model, test_loader):
    model.eval()
    correct_num, data_num = 0, 0
    for X, Y, indexes, conf_a, clean_Y in test_loader:
        for v_num in range(len(X)):
            X[v_num] = X[v_num].to(device)
        data_num += Y.size(0)
        with torch.no_grad():
            Y = Y.long().to(device)
            label = F.one_hot(Y, num_classes=classes_num)
            _, evidence_a, _ = model(X, label, 0)
            predicted = torch.argmax(evidence_a, dim=1)
            correct_num += (predicted == Y).sum().item()
    return correct_num / data_num

# torch.autograd.set_detect_anomaly(True)
args.views = len(args.dims)
args.data_path = 'datasets/' + args.dataset
samples_num = len(dataset)
classes_num = dataset.classes_num
dims = dataset.dims
index = np.arange(samples_num)
np.random.shuffle(index)
train_index, test_index = index[:int(samples_num * 0.8)], index[int(samples_num * 0.8):]
train_dataset = Subset(dataset, train_index)
test_dataset = Subset(dataset, test_index)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)  # todo
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
"""
A basic model is trained to calculate the probability of label damage
Used to add instance-level noise to clean data sets
"""
cal_model = TMC(dataset.classes_num, args.views, args.dims, args.lambda_epochs).to(device)
print(cal_model)
# print(cal_model)
optimizer = optim.Adam(cal_model.parameters(), lr=args.lr, weight_decay=1e-5)
for epoch in range(0, args.epochs):
    # print("Epoch {}".format(epoch))
    cal_model.train()
    correct_num, total_num, total_loss = 0, 0, 0.0
    for X, Y, indexes, conf, clean_Y in train_loader:
        for v_num in range(len(X)):
            X[v_num] = X[v_num].to(device)
        Y = Y.long().to(device)
        label = F.one_hot(Y, num_classes=classes_num)  
        optimizer.zero_grad()
        evidences, evidence_a, loss = cal_model(X, label, epoch)
        loss.backward()
        optimizer.step()

        pre = torch.argmax(evidence_a, dim=1)
        correct_num += torch.sum(pre == Y)
        total_num += len(Y)
        total_loss += loss.item()*len(Y)
    print("Epoch {}:  train acc: {:.3f}  test acc: {:.3f}  loss: {:.4f}".format(epoch, correct_num.item() / total_num, ModelTest(cal_model, test_loader), total_loss/total_num))

# Calculate the probability of damage for each sample
cal_model.eval()
with torch.no_grad():
    U = dict()
    X = dataset.X
    Y = torch.Tensor(dataset.Y).long().to(device)
    for v in range(args.views):
        U[v] = torch.zeros(samples_num)
        X[v] = torch.Tensor(X[v]).to(device)
    label = F.one_hot(Y, num_classes=classes_num)  
    evidences, evidence_a, _ = cal_model(X, label, 0)
    conf = {}
    for v in range(args.views):
        conf[v] = 1 - (classes_num / (torch.sum(evidences[v], dim=1) + classes_num)).cpu().numpy()
        # show_data_distribution(conf[v])
    U_a = (classes_num / (torch.sum(evidence_a, dim=1) + classes_num)).cpu().numpy()


for v in range(args.views):
    X[v] = X[v].cpu().numpy()
Y = dataset.Y
Y = np.eye(classes_num)[Y]


# Apply noise
# Flip
if args.noise_type == "flip":
    # S = 0.1 * 1 * np.array([[0, 10, 0],
    #                         [0, 0, 10],
    #                         [10, 0, 0]])

    # S = np.zeros((classes_num, classes_num))
    # S[0, 5], S[5, 0], S[10, 15], S[15, 10], S[20, 25], S[25, 20], S[30, 35], S[35, 30], S[40, 45], S[45, 40], S[50, 55], S[55, 50], S[60, 65], S[65, 60] = \
    #     10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10

    # UCI
    S = 1.1 * 1 / 2 * np.array([[0, 10, 0, 0, 0, 0, 0, 0, 0, 0],
                                [10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 10],
                                [0, 0, 0, 0, 0, 0, 0, 0, 10, 0]])

# Symmetric
elif args.noise_type == "sym":  # Here the number on the diagonal is 0, and the rest is the ratio of conversion between different classes
    # S = 0.1 * 1 / 2 * np.array([[0, 5, 10],
    #                             [15, 0, 5],
    #                             [5, 5, 0]])

    # Defines the range of non-diagonal elements
    min_value = 1
    max_value = 10
    S = np.zeros((classes_num, classes_num))
    # Assign random values to non-diagonal elements
    for i in range(classes_num):
        for j in range(classes_num):
            if i != j:  # Non-diagonal elements
                S[i, j] = np.random.uniform(min_value, max_value)

    # todo There are some problems with the treatment of S here
    # S = 0.1 * 1 / 2 * S
    row_sums = S.sum(axis=1)
    S = S / row_sums[:, np.newaxis]

def S_to_T(S, mus=0.35, classes_num=4):
    mus = 1 - mus
    T = S * mus + (1 - mus) * np.eye(classes_num)
    return T

def apply_class_noise(y, T, n_class):
    y_noisy = np.zeros_like(y)
    probas = y.dot(T)
    for i, p in enumerate(probas):
        p /= p.sum()
        k = np.random.choice(n_class, p=p)
        y_noisy[i][k] = 1
    print("Switched {}% of occurences".format(100 * (1 - (y_noisy == y).all(axis=1).mean())))
    return y_noisy

def export_dataset(dt_name, X, Y_true, Y_noisy, conf, conf_a):
    save_dir = "datasets/" + args.dataset + "/{}".format(dt_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Export
    with open("{}/X.pkl".format(save_dir), 'wb') as file:
        pickle.dump(X, file)
    with open("{}/Y_noisy.pkl".format(save_dir), 'wb') as file:
        pickle.dump(Y_noisy, file)
    with open("{}/Y_true.pkl".format(save_dir), 'wb') as file:
        pickle.dump(Y_true, file)
    with open("{}/conf.pkl".format(save_dir), 'wb') as file:
        pickle.dump(conf, file)
    with open("{}/conf_a.pkl".format(save_dir), 'wb') as file:
        pickle.dump(conf_a, file)

def apply_noise(y, conf_a, S, classes_num, evidence_a):
    y_noisy = np.zeros_like(y)
    for i, conf in enumerate(conf_a):
        T = S_to_T(S, mus=conf, classes_num=classes_num)
        p = y[i].dot(T)
        # p /= p.sum()
        # if (p < 0).any():
        #     print(conf, T, p)
        #     # p[p < 0] = 0
        # p[p < 0] = 0
        k = np.random.choice(classes_num, p=p)
        y_noisy[i][k] = 1
    print("Switched {:.4f}% of occurences".format(100 * (1 - (y_noisy == y).all(axis=1).mean())))
    return y_noisy

def apply_IDN(y, selected_indices, evidence_a):
    y_noisy = np.zeros_like(y)
    evidence_a = evidence_a.cpu().numpy()
    for i in range(len(y)):
        if i in selected_indices:
            evidence_a[i] = evidence_a[i] * (1 - y[i])
            # evidence_a[i] contains the highest class except class y[i] and is used as the noise label
            noise_class = np.argmax(evidence_a[i])
            y_noisy[i][noise_class] = 1
        else:
            y_noisy[i] = y[i]
    print("Switched {:.4f}% of occurences".format(100 * (1 - (y_noisy == y).all(axis=1).mean())))
    return y_noisy

for ratio in noise_ratios:
    print("-- Noise : {:.2f}".format(ratio))

    ############
    ## Create noise
    ############
    if args.noise_type == "flip":
        # Uniform noise  
        T = S_to_T(S, mus=1 - ratio, classes_num=classes_num)
        Y_noisy = apply_class_noise(Y, T, classes_num)

        noise = U_a * (2 * ratio)  
        conf_a = (1 - noise) * args.noise_adjust
    elif args.noise_type == "sym":
        # Non-uniform noise 
        print("U_a average:  " + str(np.mean(U_a)))
        noise = U_a * (2 * ratio) 
        conf_a = (1 - noise) * args.noise_adjust
        print("conf_a average:" + str(np.mean(conf_a)))
        Y_noisy = apply_noise(Y, conf_a, S, classes_num, evidence_a)
    elif args.noise_type == "IDN":
        nums_to_change = int(ratio * len(Y))

        weights = U_a / U_a.sum()

        # Using random selection with weights, select num_samples_to_change samples
        selected_indices = np.random.choice(len(U_a), nums_to_change, replace=False, p=weights)
        # Obtain the corresponding sample according to the selected index
        # selected_samples = [Y[i] for i in selected_indices]
        # Calculate the probability that the uncertainty of the selected sample accounts for the first ratio%
        sorted_uncertainties = np.sort(U_a)
        threshold_uncertainty = sorted_uncertainties[int(ratio * len(sorted_uncertainties))]
        probability = np.sum(U_a[selected_indices] >= threshold_uncertainty) / nums_to_change
        print("Probability of Uncertainty in Top {:.2f}%: {:.2f}%".format(ratio * 100, probability * 100))
        Y_noisy = apply_IDN(Y, selected_indices, evidence_a)

        conf_a = 1 - U_a

    export = True  # Export save or not
    if export:
        export_dataset("%s-%.2f" % (args.noise_type, ratio), X, Y, Y_noisy, conf, conf_a)

