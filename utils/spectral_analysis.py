import torch
from utils.config import config
from xitorch.linalg import symeig
from xitorch import LinearOperator


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

    # backpropagation trick
    A = (final_A - distances).detach() + distances
    # compute degree matrix
    D = torch.diag(A.sum(1))
    # compute laplacian
    L = D - A
    return A, D, L

def calc_ADL(data: torch.Tensor, sigma=1.):
    return calc_ADL_from_dist(calc_euclid_dist(data), sigma)

def find_eigs(laplacian: torch.Tensor, n_pairs:int=0, largest=False ):
    if n_pairs > 0:
        # eigenvalues, eigenvectors = torch.lobpcg(laplacian, n_pairs, largest=torch.tensor([largest]))
        # eigenvalues, eigenvectors = LOBPCG2.apply(laplacian, n_pairs)
        eigenvalues, eigenvectors = symeig(LinearOperator.m(laplacian, True), n_pairs)
    else:
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        # eigenvalues = eigenvalues.to(float)
        # eigenvectors = eigenvectors.to(float)
        sorted_indices = torch.argsort(eigenvalues, descending=largest)
        eigenvalues, eigenvectors = eigenvalues[sorted_indices], eigenvectors[:,sorted_indices]

    return eigenvalues, eigenvectors

def calc_energy_from_values(values: torch.Tensor, norm=False):
    nsamples = len(values)
    max_value = nsamples - 1 if norm else nsamples * (nsamples - 1)
    dir_energy = values.sum()
    energy_p = dir_energy / max_value
    return energy_p.cpu().item()

def normalize_A(A, D):
    inv_d = D.pow(-0.5)
    inv_d[torch.isinf(inv_d)] = 0
    # return torch.sqrt(torch.linalg.inv(D)) @ A @ torch.sqrt(torch.linalg.inv(D))
    return inv_d @ A @ inv_d

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
                       norm_lap=False, norm_eigs=False, n_pairs=0):
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
    eigenvalues, eigenvectors = find_eigs(L, n_pairs=n_pairs)
    energy = calc_energy_from_values(eigenvalues, norm=norm_lap)
    if norm_eigs and not norm_lap:
        eigenvalues = eigenvalues / (len(eigenvalues))
    return energy, eigenvalues, eigenvectors, L


class LOBPCG2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A: torch.Tensor, k:int):
        e, v = torch.lobpcg(A, k=k, largest=False)
        res = (A @ v) - (v @ torch.diag(e))
        assert (res.abs() < 1e-3).all(), 'A v != e v => incorrect eigenpairs'
        ctx.save_for_backward(e, v, A)
        return e, v

    @staticmethod
    def backward(ctx, de, dv):
        """
        solve `dA v + A dv = dv diag(e) + v diag(de)` for `dA`
        """
        e, v, A = ctx.saved_tensors

        vt = v.transpose(-2, -1)
        rhs = ((dv @ torch.diag(e)) + (v @ torch.diag(de)) - (A @ dv)).transpose(-2, -1)

        n, k = v.shape
        K = vt[:, :vt.shape[0]]
        # print('K.det=', K.det())  # should be > 0
        iK = K.inverse()

        dAt = torch.zeros((n, n), device=rhs.device)
        dAt[:k] = (iK @ rhs)[:k]
        dA = dAt.transpose(-2, -1)

        # res = T.mm(dA, v) + T.mm(A, dv) - T.mm(dv, T.diag(e)) - T.mm(v, T.diag(de))
        # print('res=', res)
        return dA, None

