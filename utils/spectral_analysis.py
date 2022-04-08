import torch
from utils.config import config

def calc_ADL_from_dist(dist_matrix: torch.Tensor, sigma=1.):
    # compute affinity matrix, heat_kernel
    A = torch.exp(-dist_matrix / (sigma ** 2))
    # compute degree matrix
    D = torch.diag(A.sum(1))
    # compute laplacian
    L = D - A
    return A, D, L

def calc_euclid_dist(data: torch.Tensor):
    return ((data.unsqueeze(0) - data.unsqueeze(1)) ** 2).sum(-1)

def calc_dist_weiss(nu: torch.Tensor, logvar: torch.Tensor):
    var = logvar.exp()
    edist = calc_euclid_dist(nu)
    wdiff = (var.unsqueeze(0) + var.unsqueeze(1) -2*(torch.sqrt(var.unsqueeze(0)*var.unsqueeze(1)))).sum(-1)
    return edist + wdiff

def calc_ADL_knn(distances: torch.Tensor, k: int, symmetric: bool = True):
    new_A = torch.clone(distances)
    mask = torch.eye(new_A.shape[0])
    new_A[mask==1] = +torch.inf

    final_A = torch.zeros_like(new_A)
    idxes = new_A.topk(k, largest=False)[1]
    final_A[torch.arange(len(idxes)).unsqueeze(1), idxes] = 1

    if symmetric:
        final_A = ((final_A + final_A.T) > 0).float()
        # final_A = 0.5*(final_A + final_A.T)

    # compute degree matrix
    D = torch.diag(final_A.sum(1))
    # compute laplacian
    L = D - final_A
    return final_A, D, L

def calc_ADL(data: torch.Tensor, sigma=1.):
    return calc_ADL_from_dist(calc_euclid_dist(data), sigma)

def find_eigs(laplacian: torch.Tensor):
    eigenvalues, eigenvectors = torch.linalg.eig(laplacian)
    eigenvalues = eigenvalues.to(float)
    eigenvectors = eigenvectors.to(float)
    sorted_indices = torch.argsort(eigenvalues)
    eigenvalues, eigenvectors = eigenvalues[sorted_indices], eigenvectors[:,sorted_indices]
    return eigenvalues, eigenvectors

def calc_energy_from_values(values: torch.Tensor, norm=False):
    nsamples = len(values)
    max_value = nsamples - 1 if norm else nsamples * (nsamples - 1)
    dir_energy = values.sum()
    energy_p = dir_energy / max_value
    return energy_p.cpu().item()

def normalize_A(A, D):
    return torch.sqrt(torch.linalg.inv(D)) @ A @ torch.sqrt(torch.linalg.inv(D))

def dir_energy_normal(data: torch.Tensor, sigma=1.):
    A, D, L = calc_ADL(data, sigma)
    L_norm = torch.eye(A.shape[0]).to(config.device) - normalize_A(A, D)
    eigenvalues, eigenvectors = find_eigs(L_norm)
    energy = calc_energy_from_values(eigenvalues, norm=True)
    return energy, eigenvalues, eigenvectors

def dir_energy(data: torch.Tensor, sigma=1):
    A, D, L = calc_ADL(data, sigma=sigma)
    eigenvalues, eigenvectors = find_eigs(L)
    energy = calc_energy_from_values(eigenvalues)
    return energy

def laplacian_analysis(data: torch.Tensor, sigma=1., knn=0, logvars: torch.Tensor=None,
                       norm_lap=False, norm_eigs=False):
    if logvars is None:
        distances = calc_euclid_dist(data)
    else:
        distances = calc_dist_weiss(data, logvars)
    if knn > 0:
        A, D, L = calc_ADL_knn(distances, knn, symmetric=True)
    else:
        A, D, L = calc_ADL_from_dist(distances, sigma)
    if norm_lap:
        L = torch.eye(A.shape[0]).to(config.device) - normalize_A(A, D)
    eigenvalues, eigenvectors = find_eigs(L)
    energy = calc_energy_from_values(eigenvalues, norm=norm_lap)
    if norm_eigs and not norm_lap:
        eigenvalues = eigenvalues / (len(eigenvalues))
    return energy, eigenvalues, eigenvectors, L