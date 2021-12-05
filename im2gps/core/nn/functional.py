import numpy as np
from im2gps.core.nn.enum import NNEnum
import math
import torch
import torch.nn.functional as f


def l2_distance_from_query(q, neighbours):
    if len(q.shape) == 2:
        q = q.unsqueeze(1)
        if len(neighbours.shape) == 2:
            neighbours = neighbours.unsqueeze(1)
        elif len(neighbours.shape) != 3:
            raise ValueError(f"When q is 2D tensor neighbours should be 2D or 3D tensor, "
                             f"actual shape {list(neighbours.shape)}")
        return torch.cdist(q, neighbours).squeeze(1)
    elif len(q.shape) == 1:
        q = q.unsqueeze(0)
        if len(neighbours.shape) == 1:
            neighbours = neighbours.unsqueeze(0)
        elif len(neighbours.shape) != 2:
            raise ValueError(f"When q is 1D tensor neighbours should be 1D or 2D tensor, "
                             f"actual shape {list(neighbours.shape)}")
        return torch.cdist(q, neighbours).squeeze()
    else:
        raise ValueError(f"q should be 1D or 2D tensor, actual shape {list(q.shape)}")


def cos_distance_from_query(q, neighbours):
    if len(q.shape) == 2:
        if len(neighbours.shape) == 3:
            return f.cosine_similarity(q.unsqueeze(1), neighbours, dim=2)
        elif len(neighbours.shape) == 2:
            return f.cosine_similarity(q, neighbours).unsqueeze(1)
        else:
            raise ValueError(f"When q is 2D tensor neighbours should be 2D or 3D tensor, "
                             f"actual shape {list(neighbours.shape)}")
    elif len(q.shape) == 1:
        if len(neighbours.shape) == 2:
            return f.cosine_similarity(q.unsqueeze(0), neighbours)
        elif len(neighbours.shape) == 1:
            return f.cosine_similarity(q, neighbours, dim=0)
        else:
            raise ValueError(f"When q is 1D tensor neighbours should be 1D or 2D tensor, "
                             f"actual shape {list(neighbours.shape)}")
    else:
        raise ValueError(f"q should be 1D or 2D tensor, actual shape {list(q.shape)}")


def distance_from_query(q: torch.Tensor, neighbours: torch.Tensor, dist_type=NNEnum.L2_DIST):
    """
    Compute distance from query descriptors to neighbours.
    :param q: Tensor with query. Can be of size BxD or D.
    :param neighbours: Tensor with neighbours descriptors. Can be BxNxD or BxD when q is BxD and NxD or D when q is D.
    :param dist_type: type of distance to compute
    :return: Returns distance from query to each neighbour in batch
             1) When q is BxD and neighbours is BxNxD returns tensor of size BxN,
             2) When q is BxD and neighbours is BxD returns tensor of size Bx1,
             3) When q is D and neighbours is NxD returns tensor of size N,
             4) When q is D and neighbours is D returns tensor of size 1
    """
    if dist_type == NNEnum.L2_DIST:
        return l2_distance_from_query(q, neighbours)
    elif dist_type == NNEnum.COS_DIST:
        return cos_distance_from_query(q, neighbours)
    else:
        raise ValueError(f"Unknown dist_type: {dist_type}")


def dist2weights(x: torch.Tensor, m, dist_type=NNEnum.L2_DIST, eps=1e-8):
    if dist_type == NNEnum.L2_DIST:
        return (1 / (x + eps)) ** m
    elif dist_type == NNEnum.COS_DIST:
        return (x + 1) ** m
    else:
        raise ValueError(f'Unknown distance type: {dist_type}')


def multivariate_normal_pdf(x: torch.Tensor, mu: torch.Tensor, cov: torch.Tensor):
    """

    :param x: BxQxD
    :param mu: BxNxD
    :param sigma: int or DxD
    :return: BxQxN
    """

    assert len(x.shape) == len(mu.shape), f"mu {list(mu.shape)} and x {list(x.shape)} must have same number of " \
                                          f"dimensions"
    assert len(x.shape) == 2 or len(x.shape) == 3, f"mu {list(mu.shape)} and x {list(x.shape)} should be 2D or 3D " \
                                                   f"tensors"
    if len(x.shape) == 3:
        assert x.shape[0] == mu.shape[0], f"mu {list(mu.shape)} and x {list(x.shape)} must have same batch size"

    assert x.shape[-1] == mu.shape[-1], f"mu {list(mu.shape)} and x {list(x.shape)} must have same last dimension size"

    k = x.shape[-1]
    det = cov.det()
    inv = torch.inverse(cov)

    if len(x.shape) == 2:
        centered = x.unsqueeze(1) - mu
        res1 = torch.einsum('qnd, dj -> qnj', centered, inv)
        exponent = torch.einsum('qnd, qnd -> qn', res1, centered)
    elif len(x.shape) == 3:
        centered = x.unsqueeze(2) - mu.unsqueeze(1)
        res1 = torch.einsum('bqnd, dj -> bqnj', centered, inv)
        exponent = torch.einsum('bqnd, bqnd -> bqn', res1, centered)
    else:
        raise ValueError(f"x {list(x.shape)} should be a 2D or 3D tensor")

    pi = torch.tensor(math.pi)
    return torch.pow(2 * pi, -k / 2) * torch.pow(det, -0.5) * torch.exp(-0.5 * exponent)


def kde(weights: torch.Tensor, neighbour_coords, coord_space, sigma):
    """
    Compute weighted kernel density estimation and find data point with maximal probability

    :param weights: Weights corresponding to each data point.Tensor of size BxN or N, where B is batch size and N
    is number of data points.
    :param neighbour_coords: Coordinates that represent each data point. Tensor of size BxNx2, where B is batch size and
    N is is number of data points.
    :param coord_space: BxQx2 points over which to perform KDE.
    :param sigma: Standard deviation of normal distribution
    :return: coordinates with maximal probability estimated using KDE
    """
    cov = torch.eye(neighbour_coords.shape[-1]).cuda() * torch.abs(sigma)
    pdfs = multivariate_normal_pdf(coord_space, neighbour_coords, cov)  # BxQxN
    if len(pdfs.shape) == 2:
        weighted_pdfs = torch.einsum('n,mn -> mn', weights, pdfs)
        pdf = torch.einsum('mn->m', weighted_pdfs)
        normalize_by = 1 / (weights.sum() * torch.pow(cov.det(), 0.5))
        return pdf / normalize_by
    elif len(pdfs.shape) == 3:
        weighted_pdfs = torch.einsum('bn,bmn -> bmn', weights, pdfs)  # BxQxN
        pdf = torch.einsum('bmn->bm', weighted_pdfs)  # BxN
        normalize_by = 1 / (weights.sum(dim=1) * torch.pow(cov.det(), 0.5))
        normalize_by = normalize_by.unsqueeze(0).t()
        return pdf / normalize_by
    else:
        raise ValueError(f"Wrong dimension of pdfs {list(pdfs.shape)}")


def haversine_loss(predicted, ground_true, units=NNEnum.HAV_KILOMETERS):
    """
    :param predicted: tensor of size Bx2 or 2, first element is longitude, second element is latitude
    :param ground_true: tensor of size Bx2 or 2, first element is longitude, second element is latitude
    :param units: type of units to use
    :return:
    """
    return torch.mean(haversine_distance(predicted, ground_true, units))


def _haversine(lng1, lat1, lng2, lat2, units):
    radius = torch.tensor(6371.0088)  # Earth radius
    lng1_rad = torch.deg2rad(lng1)
    lat1_rad = torch.deg2rad(lat1)

    lng2_rad = torch.deg2rad(lng2)
    lat2_rad = torch.deg2rad(lat2)

    hav_lng = torch.sin((lng2_rad - lng1_rad) * 0.5) ** 2
    hav_lat = torch.sin((lat2_rad - lat1_rad) * 0.5) ** 2

    x = hav_lat + torch.cos(lat1_rad) * torch.cos(lat2_rad) * hav_lng

    if units == NNEnum.HAV_KILOMETERS:
        return 2 * radius * torch.asin(torch.sqrt(x))
    elif units == NNEnum.HAV_METERS:
        return 2 * radius * torch.asin(torch.sqrt(x)) * 1000


def haversine_distance(x, y, units=NNEnum.HAV_KILOMETERS):
    if len(x.shape) == 2 and len(y.shape) == 2:
        lng1 = x[:, 0]
        lat1 = x[:, 1]

        lng2 = y[:, 0]
        lat2 = y[:, 1]
    elif len(x.shape) == 1 and len(y.shape) == 1:
        lng1 = x[0]
        lat1 = x[1]

        lng2 = y[0]
        lat2 = y[1]
    else:
        raise ValueError("Dimension error")

    return _haversine(lng1, lat1, lng2, lat2, units)


def haversine_distance_over_space(coordinate_space, true_coords, units=NNEnum.HAV_KILOMETERS):
    """
    :param coordinate_space: BxNx2
    :param true_coords: Bx2
    :param units: units to use
    """
    assert len(coordinate_space.shape) == 3 and len(true_coords.shape) == 2, \
        "coordinate_space and true_coords should have batch dimension"
    lng1 = coordinate_space[:, :, 0]
    lat1 = coordinate_space[:, :, 1]

    lng2 = true_coords[:, 0].unsqueeze(1)
    lat2 = true_coords[:, 1].unsqueeze(1)

    return _haversine(lng1, lat1, lng2, lat2, units)


def l2_normalization(x, eps=1e-6):
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)


def accuracy(predicted, target):
    y = torch.argmax(predicted, dim=1)
    return torch.sum(y == target) / target.shape[0]


def get_target_index(coordinate_space, true_coords):
    dists = haversine_distance_over_space(coordinate_space, true_coords)
    return torch.argmin(dists, dim=1)


def kde_loss(pdf, target):
    return torch.nn.functional.nll_loss(pdf, target)


def generate_grid(centers: torch.Tensor, sigma, bw, n):
    """
    :param centers: BxNx2
    :param sigma:
    :param bw:
    :param n:
    :return: BxQx2 array, where Q is N*(2*bw*n+1)*(2*bw*n+1)
    """

    centers_np = centers.numpy()

    top_left = centers_np - (bw * sigma, -bw * sigma)
    x_space = np.linspace(top_left[:, :, 0], top_left[:, :, 0] + (2 * bw * sigma), num=2 * bw * n + 1)
    x_space = np.einsum('ibn->bni', x_space)  # just change order of dimensions to (batch,neighbour,i-th point)
    y_space = np.linspace(top_left[:, :, 1], top_left[:, :, 1] - (2 * bw * sigma), num=2 * bw * n + 1)
    y_space = np.einsum('ibn->bni', y_space)

    batch_size = centers_np.shape[0]

    grid_space = []
    for batch in range(batch_size):
        batch_grid_space = []
        for x, y in zip(x_space[batch], y_space[batch]):
            xx, yy = np.meshgrid(x, y)
            batch_grid_space.append(np.vstack([xx.ravel(), yy.ravel()]).T)
        grid_space.append(np.concatenate(batch_grid_space))

    return torch.tensor(np.array(grid_space), dtype=torch.float32)
