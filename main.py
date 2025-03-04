import os
import pickle
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
from data import MultiViewDataset
import argparse
from model import TMNR
import numpy as np
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, help='input batch size for training', default=128)
parser.add_argument('--epochs', type=int, help='number of epochs to train', default=100)
parser.add_argument('--lambda_epochs', type=int, help='gradually increase the value of lambda from 0 to 1', default=100)
parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
parser.add_argument('--dataset', type=str, help='PIE, Caltech101, UCI, BBC, Leaves', default="PIE")  # Due to file size limitations, we have only uploaded the PIE, Leaves, and UCI datasets here
parser.add_argument('--noise_type', type=str, help="sym, flip, IDN", default="IDN")
args = parser.parse_args()

# default
noise_ratios = np.arange(0.0, 0.55, 0.10)
args.k = 5
args.lamb = 0.01  # denotes the hyperparameter \beta in equation 14 of the article
args.gamma = 10000  # denotes the hyperparameter \gamma in equation 14 of the article
args.epochs = 100

if args.dataset == "UCI":
    args.batch_size = 128
    args.dims = [[6], [240], [47]]
    args.gamma = 10000
elif args.dataset == "PIE":
    args.batch_size = 64
    args.dims = [[484], [256], [279]]
    args.gamma = 10000
elif args.dataset == "Caltech101":
    args.batch_size = 128
    args.dims = [[48], [40], [254], [1984], [512], [928]]
    args.gamma = 1000
elif args.dataset == "Leaves":
    args.batch_size = 128
    args.dims = [[64], [64], [64]]
    args.gamma = 10000
elif args.dataset == "BBC":
    args.batch_size = 128
    args.dims = [[4659], [4633], [4665], [4684]]
    args.gamma = 10000


args.lambda_epochs = args.epochs
args.data_path = './Datasets/' + args.dataset
args.views = len(args.dims)
classes_num = 0


def main():
    global args

    def experiment(ratio, repeat_num):
        global classes_num, matrix2X
        print("======================================\nCurrent noise ratio: {}".format(ratio))

        train_loader, test_loader, classes_num, Y_true, Y_noisy = import_and_load_data(ratio)

        """
        Similarity maps are computed for each view
        """
        sample_num = len(Y_true)
        Y_noisy_int = np.argmax(Y_noisy, axis=1)
        train_dataset = train_loader.dataset
        X = train_dataset.dataset.X.copy()
        similarity_matrix = torch.zeros(args.views, sample_num, sample_num).to(device)
        for v in range(args.views):
            X[v] = torch.Tensor(X[v]).to(device)
            distance_squared = (torch.cdist(X[v], X[v], p=2) ** 2)
            similarity_matrix[v] = torch.exp(-distance_squared / 2)

        """
         Find the index of the most similar sample with the same label within each viewpoint of each sample
        """
        k = args.k
        nearest_indices = torch.zeros((args.views, sample_num, k), dtype=torch.long, device=device)
        for i in range(args.views):
            for j in range(sample_num):
                sorted_indices = torch.argsort(similarity_matrix[i][j], descending=True)
                same_label_indices = sorted_indices[torch.Tensor(Y_noisy_int).to(device)[sorted_indices] == Y_noisy_int[j]]
                # Take the k-nearest elements other than itself, and if there are less than k, fill them up with the last element.
                if k+1 <= same_label_indices.shape[0]:
                    nearest_indices[i, j, :k] = same_label_indices[1:k+1]
                else:
                    nearest_indices[i, j, :same_label_indices.shape[0]-1] = same_label_indices[1:]
                    nearest_indices[i, j, same_label_indices.shape[0] - 1:] = same_label_indices[-1]

        """
        Train the clean model
        """
        history = []
        start_time = time.time()
        model = TMNR(classes_num, args.views, args.dims, sample_num, args.lambda_epochs, similarity_matrix, nearest_indices)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        model.to(device)
        # print(model)

        for epoch in range(0, args.epochs):
            model.train()
            total_loss = 0
            for X, Y, indexes, clean_Y in train_loader:
                Y = Y.long().to(device)
                indexes.to(device)
                for v_num in range(len(X)):
                    X[v_num] = X[v_num].to(device)
                evidences, evidence_a, loss = model(X, Y, indexes, epoch, args.lamb, args.gamma)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # such that the updated matrix satisfies its own constraints
                t = model.matrixes.detach()
                t = torch.clamp(t, min=0)
                normalized_matrixes = t / t.sum(dim=-1, keepdim=True)
                with torch.no_grad():
                    model.matrixes.data = normalized_matrixes

                total_loss += loss * len(Y)
                del evidences, evidence_a
                torch.cuda.empty_cache()
            acc = test(model, test_loader, epoch)
            print('Epoch {} ====> test_acc: {:.4f}, loss = {}'.format(epoch, acc,
                                                                      total_loss / len(train_loader.dataset)))
            history.append(acc)
        print("clean_model train time:", round(time.time() - start_time, 2), "s")

    for repeat in range(0, 1):  # Control total number of runs
        for ratio in noise_ratios:
            experiment(ratio, repeat)


def import_and_load_data(mu):
    # load data
    with open(args.data_path + '/' + args.noise_type + "-{:.2f}".format(mu) + "/X.pkl", 'rb') as file:
        X = pickle.load(file)
    with open(args.data_path + '/' + args.noise_type + "-{:.2f}".format(mu) + "/Y_true.pkl", 'rb') as file:
        Y_true = pickle.load(file)
    with open(args.data_path + '/' + args.noise_type + "-{:.2f}".format(mu) + "/Y_noisy.pkl", 'rb') as file:
        Y_noisy = pickle.load(file)

    dataset = MultiViewDataset(args.dataset, X, Y_noisy, Y_true)
    samples_num = Y_noisy.shape[0]
    classes_num = Y_noisy.shape[1]
    index = np.arange(samples_num)
    np.random.shuffle(index)
    train_index, test_index = index[:int(samples_num * 0.8)], index[int(samples_num * 0.8):]
    train_dataset = Subset(dataset, train_index)
    test_dataset = Subset(dataset, test_index)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_index), shuffle=False)
    return train_loader, test_loader, classes_num, Y_true, Y_noisy


def test(model, test_loader, epoch):
    global classes_num
    model.eval()
    correct_num, data_num = 0, 0

    for X, Y, indexes, clean_Y in test_loader:
        for v_num in range(len(X)):
            X[v_num] = X[v_num].to(device)
        data_num += Y.size(0)
        with torch.no_grad():
            Y = Y.long().to(device)
            clean_Y = clean_Y.to(device)
            _, evidence_a, _ = model(X, Y, indexes, 0)
            predicted = torch.argmax(evidence_a, dim=1)
            clean_Y_ind = torch.argmax(clean_Y, dim=1)
            correct_num += (predicted == clean_Y_ind).sum().item()
    return correct_num / data_num


if __name__ == "__main__":
    main()
