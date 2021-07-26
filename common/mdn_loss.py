from __future__ import absolute_import, division

import numpy as np

from common.camera import camera_to_world, world_to_camera
from common.h36m_dataset import h36m_cameras_extrinsic_params as cams

import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent


def get_best_p1_pred(mu, target, pose_level=True):
    """
    Function to return the best of K kernels, relative to
    targets. Best kernel here represents the kernel with
    minimal distance (based on x,y,z coordinates) to the
    target

    mu: batch_size x n_nodes x n_gaussians x 3
    target: batch_size x n_nodes x 3
    pose_level: if False chooses best nodes independently, otherwise the best whole pose

    returns: the prediction per node that minimizes the distance to ground truth
    """
    num_gaussian = mu.shape[2]
    target = target.unsqueeze(2).expand(-1, -1, num_gaussian, -1)
    norm = torch.norm(mu - target, dim=3)
    if pose_level:
        pose_norm = torch.sum(norm, dim=1, keepdim=True)
        best_kernel_idx = torch.min(pose_norm, dim=2)[1]
        best_pred = mu.gather(dim=2, index=best_kernel_idx.unsqueeze(2).unsqueeze(3).repeat([1,mu.shape[1],1,3])).squeeze()
    else:    
        best_kernel_idx = torch.min(norm, dim=2)[1]
        best_pred = mu.gather(dim=2, index=best_kernel_idx.unsqueeze(2).unsqueeze(3).repeat([1,1,1,3])).squeeze()
    return best_pred

def best_p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers. Chooses the best per pose
    
    Args:
    mu: batch_size x n_nodes x n_gaussians x 3
    target: batch_size x n_nodes x 3

    Returns the mean error of best pred after alignment
    """
    num_gaussian = predicted.shape[2]
    predicted = predicted.transpose([0, 2, 1, 3])
    
    target = np.expand_dims(target,1)
    target = np.tile(target, [1, num_gaussian, 1, 1])

    muX = np.mean(target, axis=2, keepdims=True)
    muY = np.mean(predicted, axis=2, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(2, 3), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(2, 3), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 1, 3, 2), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 1, 3, 2)
    R = np.matmul(V, U.transpose(0, 1, 3, 2))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=2))
    V[:, :, :, -1] *= sign_detR
    s[:, :,  -1] *= sign_detR.squeeze(axis=2)#.flatten()
    R = np.matmul(V, U.transpose(0, 1, 3, 2))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=2, keepdims=True), axis=3)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return np.mean(np.min(np.mean(np.linalg.norm(predicted_aligned - target, axis=3),axis=2), axis=1))

def sample_predict(outs, target, pose_level=True, k=50):
    '''
    Function to predict a sample of the distribution

    outs: (mu, sigma, pi)
    mu - [batch_size, n_nodes, n_gaussians, 3]
    sigma - [batch_size, n_nodes, n_gaussians, 1] after activation function
    pi - [batch_size, n_nodes, n_gaussians] before activation
    '''
    _mu = outs[0]
    _sigma = outs[1]
    _pi = outs[2]

    m = Independent(Normal(loc=_mu, scale=_sigma), 1)
    
    #print(preds.shape)
    if pose_level:
        _pi = torch.mean(_pi, dim=1)
        pi = torch.max(F.softmax(_pi, dim=1), 1e-10*torch.ones([1]))
        kernel_d = torch.distributions.Categorical(pi)

        preds = []
        for _ in range(k):
            pred_init = m.sample()
            kernel_idx = kernel_d.sample()
            #print(kernels.shape) [batch_size]
            pred = pred_init.gather(dim=2, index=kernel_idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat([1,_mu.shape[1],1,3])).squeeze()
            preds.append(pred)
    else:
        pi = torch.max(F.softmax(_pi, dim=1), 1e-10*torch.ones([1]))
        kernel_d = torch.distributions.Categorical(pi)

        preds = []
        for _ in range(k):
            pred_init = m.sample()
            kernel_idx = kernel_d.sample()
            #print(kernel_idx) [batch_size, n_nodes]
            pred = pred_init.gather(dim=2, index=kernel_idx.unsqueeze(2).unsqueeze(3).repeat([1,1,1,3])).squeeze()
            preds.append(pred)

    pred = torch.cat([p.unsqueeze(2) for p in preds], dim=2).cpu()
    out = {}
    for n in [1, 5, 20, 50]:
        out['{}_samples_avg'.format(n)] = torch.mean(pred[:, :, :n, :], dim=2)
        out['{}_samples_best'.format(n)] = get_best_p1_pred(pred[:, :, :n, :], target)
    return out

def sample_kernels(outs, target, k=200):
    '''
    Function to predict a sample of the kernels

    outs: (mu, sigma, pi)
    mu - [batch_size, n_nodes, n_gaussians, 3]
    sigma - [batch_size, n_nodes, n_gaussians, 1] after activation function
    pi - [batch_size, n_nodes, n_gaussians] before activation
    '''
    _mu = outs[0]
    _sigma = outs[1]
    _pi = outs[2]
    
    pi = torch.max(F.softmax(_pi, dim=1), 1e-10*torch.ones([1]))
    kernel_d = torch.distributions.Categorical(pi)

    preds = []
    for _ in range(k):
        kernel_idx = kernel_d.sample()
        #print(kernel_idx) [batch_size, n_nodes]
        pred = _mu.gather(dim=2, index=kernel_idx.unsqueeze(2).unsqueeze(3).repeat([1,1,1,3])).squeeze()
        preds.append(pred)
    pred = torch.cat([p.unsqueeze(2) for p in preds], dim=2).cpu()
    out = {}
    for n in [1, 5, 20, 50, 200]:
        out['{}_samples_avg'.format(n)] = torch.mean(pred[:, :, :n, :], dim=2)
        out['{}_samples_best'.format(n)] = get_best_p1_pred(pred[:, :, :n, :], target)
    return out

def mdn_loss_fn(outs, y, pose_level=False):
    '''
    Function to calculate the loss function, given a set
    of Gaussians (mean of location) outs, and targets (y)

    outs: (mu, sigma, pi)
    mu - [batch_size, n_nodes, n_gaussians, 3]
    sigma - [batch_size, n_nodes, n_gaussians, 1] after activation function
    pi - [batch_size, n_nodes, n_gaussians] before activation
    y: [batch_size, n_nodes, 3]
    pose_level: whether to use pose level distributions or node levek
    '''
    _mu = outs[0]
    _sigma = outs[1]
    _pi = outs[2]

    y = y.unsqueeze(dim=2)

    m = Independent(Normal(loc=_mu, scale=_sigma), 1)
    loss = m.log_prob(y)

    if pose_level:
        _pi = torch.mean(_pi, dim=1)
        pi = torch.max(F.softmax(_pi, dim=1), 1e-10*torch.ones([1]))
        loss = m.log_prob(y)
        loss = torch.sum(loss, dim=1, keepdim=False)
        loss = -torch.logsumexp(loss + torch.log(pi), dim=1, keepdim=True)
    else:
        pi = torch.max(F.softmax(_pi, dim=2), 1e-10*torch.ones([1]))
        loss = torch.logsumexp(loss + torch.log(pi), dim=2)
        loss = -torch.sum(loss, dim=1, keepdim=True)

    return torch.mean(loss)

def mdn_loss_fn_pose_distributions(outs, y):
    '''
    wrapper
    '''
    return mdn_loss_fn(outs, y, pose_level=True)

def get_max_pi_predictions(means, logits):
    '''
    Given a set of kernels, return a single prediction
    per joint, based on the max pi (mixing coefficient)
    from set of K distributions.

    logits: [batch_size, n_nodes, n_gaussians]
    '''

    preds = torch.argmax(logits, dim=2, keepdim=True)
    preds = means.gather(dim=2, index=preds.unsqueeze(2).repeat([1,1,1,3])).squeeze()
    return preds

def get_avg_predictions(means, logits):
    '''
    Given a set of kernels, return a single prediction
    per joint, based on the average of the set of
    K distributions. Average for each joint is calculated
    as the mean of the kernel positions weighted by their
    mixing coefficients.

    logits: [batch_size, n_nodes, n_gaussians]
    '''

    pi = torch.max(F.softmax(logits, dim=2), 1e-10*torch.ones([1]))
    preds = torch.sum(pi.unsqueeze(3) * means, dim=2)
    return preds

def get_aligned_preds(outs, subj):
    mu = outs[0]
    logits = outs[2]

    num_gaussian = mu.shape[2]
    #_mu = get_max_pi_predictions(mu, logits)
    _mu = get_avg_predictions(mu, logits)

    vals = []
    #for k in range(5):
    tmp = []
    for c in range(4):
        #vals.append(torch.tensor(camera_to_world(mu[c,:,k,:].cpu().double(), cams[subj[c]][c]['orientation'], t=0)))
        vals.append(torch.tensor(camera_to_world(_mu[c,:,:].cpu().double(), cams[subj[c]][c]['orientation'], t=0)))
        #vals.append(tmp)

    mean = torch.mean(torch.stack(vals), dim=0)
    world = []
    for c in range(4):
        world.append(torch.tensor(world_to_camera(mean.cpu(),
                                                  np.array(cams[subj[c]][c]['orientation']),
                                                  t=0)))
    world = torch.stack(world).cpu()
    #pred_best = get_best_p1_pred(mu.cpu().double(), world.unsqueeze(2).expand(-1, -1, num_gaussian, -1), pose_level=False)
    #preds_best = mu.cpu().gather(dim=2, index=best_kernel_idx.unsqueeze(2).unsqueeze(3).repeat([1, 1, 1, 3])).squeeze()
    return world.float()
    #return preds_best.cpu()

def get_all_preds(outs, targets=None, aligned=False, subj=None):
    '''
    Return a set of max, avg, best and possibly aligned
    kernel predictions and possible preds chosen by discriminator
    '''
    mu = outs[0]
    pi_logits = outs[2]

    num_gaussian = mu.shape[2]

    preds_max = get_max_pi_predictions(mu, pi_logits)
    preds_mean = get_avg_predictions(mu, pi_logits) 

    if targets is not None:
        preds_best = get_best_p1_pred(mu.cpu(), targets)
    else:
        preds_best = preds_mean

    return_dict = {'max': preds_max.cpu(), 'mean': preds_mean.cpu(), 'best': preds_best.cpu()}

    if aligned:
        preds_align = get_aligned_preds(outs, subj)
        return_dict['align'] = preds_align
    
    #sample_outs = sample_kernels(outs, targets)#, pose_level=False)
    #return_dict.update(sample_outs)

    return return_dict


    
    
    
