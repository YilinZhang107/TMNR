import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


# Adjusted cross-entropy loss based on the Dirichlet distribution
def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    # label = F.one_hot(p, num_classes=c)
    label = p
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return A + B


# Similarity-based constrained noise transfer matrix
def similarity_loss(S, near_indices, matrix, indexes):
    k = near_indices.shape[-1]
    classes = matrix.shape[-1]
    bs = len(indexes)
    x_indices = indexes.long()
    y_indices = near_indices[x_indices]  # Its nearest k neighbors

    m1 = matrix[x_indices]

    m1 = m1.unsqueeze(1).expand(bs, k, classes, classes)
    m2 = matrix[y_indices]

    # Only non-diagonal elements are counted, so that all diagonal elements are zero.
    m1 = m1 * (1 - torch.eye(classes, device=device).unsqueeze(0).unsqueeze(0))
    m2 = m2 * (1 - torch.eye(classes, device=device).unsqueeze(0).unsqueeze(0))

    x_indices = x_indices.unsqueeze(1).expand(bs, k)
    S_values = S[x_indices, y_indices]  # bs * k
    # Calculate the loss of similarity for each pair of samples
    distance = torch.norm(m1 - m2 + 1e-10, dim=(2, 3))  # **2
    pairwise_loss = S_values * distance

    loss = torch.sum(pairwise_loss, dim=1, keepdim=True) / k
    return loss


# Based on the loss of confidence, confidence and t_ii should be proportional to each other
def conf_loss(conf_a, T, indexes, y, class_num):

    # Calculate the mean uncertainty value for all categories
    classes_conf = torch.zeros(class_num, device=device)
    class_counts = torch.bincount(y, minlength=class_num).float()
    classes_conf = torch.scatter_add(classes_conf, dim=0, index=y, src=conf_a)
    classes_conf = classes_conf / (class_counts+1e-5)

    # Get diagonal elements
    diagonal_elements = torch.diagonal(T[indexes], offset=0, dim1=-2, dim2=-1)
    # Create a one-hot coding matrix representing the true category of each sample
    one_hot_y = torch.zeros(len(conf_a), class_num, device=device)
    one_hot_y.scatter_(1, y.view(-1, 1), 1)
    loss = ((conf_a.view(-1, 1) - diagonal_elements) ** 2 * one_hot_y).sum(dim=1) + \
           ((classes_conf.view(1, -1) - diagonal_elements) ** 2 * (1 - one_hot_y)).sum(dim=1)

    return loss.reshape(-1, 1)


# Loss of inter-view consistency
def consistent_view_loss(T, indexes):
    T = T[:, indexes]

    bs = T.shape[1]
    v = T.shape[0]
    K = T.shape[2]

    reshaped_T = T.view(v, bs, -1)
    # Calculate the squared differences between all pairs of matrices
    squared_diff = (reshaped_T.unsqueeze(0) - reshaped_T.unsqueeze(1)).pow(2)
    # Apply mask to exclude self-comparisons and sum the squared differences
    mask = torch.eye(v, v, device=device).bool()
    squared_diff_masked = squared_diff[~mask].view(v, v - 1, bs, -1)
    loss_optimized = squared_diff_masked.sum(dim=-1).sum(dim=(0, 1)) / 2 / (K*K)
    # print("viewMatrix_loss_optimized time: ", time.time() - start)

    return loss_optimized.view(-1, 1)










