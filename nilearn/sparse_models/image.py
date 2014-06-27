# -*- coding: utf-8 -*-
import numpy as np

def div(grad):
    """ Compute divergence of image gradient """
    res = np.zeros(grad.shape[1:])
    for d in range(grad.shape[0]):
        this_grad = np.rollaxis(grad[d], d)
        this_res = np.rollaxis(res, d)
        this_res[:-1] += this_grad[:-1]
        this_res[1:-1] -= this_grad[:-2]
        this_res[-1] -= this_grad[-2]
    return res


def grad(img):
    """ Compute gradient of an image

        Parameters
        ===========
        img: ndarray
            N-dimensional image

        Returns
        =======
        gradient: ndarray
            Image of the gradient: the i-th component along the first
            axis is the gradient along the i-th axis of the original
            array img
    """
    shape = [img.ndim, ] + list(img.shape)
    gradient = np.zeros(shape, dtype=img.dtype)
    # 'Clever' code to have a view of the gradient with dimension i stop
    # at -1
    slice_all = [0, slice(None, -1),]
    for d in range(img.ndim):
        gradient[slice_all] = np.diff(img, axis=d)
        slice_all[0] = d+1
        slice_all.insert(1, slice(None))
    return gradient

def grad_with_mask(M, mask):
    """
    Compute gradient using mask (2D or 3D)
    """
    ind = indices_for_grad(mask)
    return grad_from_indices(M, ind)

def div_with_mask(P, mask):
    """
    Compute divergence using mask (2D or 3D)
    """
    ind = indices_for_div(mask)
    return div_from_indices(P, ind)

def indices_for_grad(mask):
    """
    Compute indices for the gradient
    return a list of (grad_0,grad_1,grad_2)
    grad is compute as :
    grad = np.zeros_like(X)
    grad[grad_x[1]] = X[grad_x[0]]-X[grad_x[1]]
    """
    ### Construction of the indices matrix
    M_ = mask.astype('int')
    M_ind_ = np.zeros_like(M_)
    M_ind_[mask] = range(np.sum(mask))

    if np.size(mask.shape) == 2 :
        # For axe x
        grad_x = []
        fx_ = np.zeros_like(M_)
        fx_[1:,:] = 2*M_[1:,:] - M_[:-1,:]
        ind_ = (fx_[:,:]==1)
        ind__ = np.zeros_like(ind_)
        ind__[:-1,:] = ind_[1:,:]
        gi = np.zeros(np.sum(mask),dtype=bool)
        gi[M_ind_[ind_]] = True
        grad_x.append(gi)
        gi = np.zeros(np.sum(mask),dtype=bool)
        gi[M_ind_[ind__]] = True
        grad_x.append(gi)

        # For axe y
        grad_y = []
        fy_ = np.zeros_like(M_)
        fy_[:,1:] = 2*M_[:,1:] - M_[:,:-1]
        ind_ = (fy_[:,:]==1)
        ind__ = np.zeros_like(ind_)
        ind__[:,:-1] = ind_[:,1:]
        gi = np.zeros(np.sum(mask),dtype=bool)
        gi[M_ind_[ind_]] = True
        grad_y.append(gi)
        gi = np.zeros(np.sum(mask),dtype=bool)
        gi[M_ind_[ind__]] = True
        grad_y.append(gi)

        return [grad_x,grad_y]

    if np.size(mask.shape) == 3 :
        # For axe x
        grad_x = []
        fx_ = np.zeros_like(M_)
        fx_[1:,:,:] = 2*M_[1:,:,:] - M_[:-1,:,:]
        ind_ = (fx_[:,:,:]==1)
        ind__ = np.zeros_like(ind_)
        ind__[:-1,:,:] = ind_[1:,:,:]
        gi = np.zeros(np.sum(mask),dtype=bool)
        gi[M_ind_[ind_]] = True
        grad_x.append(gi)
        gi = np.zeros(np.sum(mask),dtype=bool)
        gi[M_ind_[ind__]] = True
        grad_x.append(gi)

        # For axe y
        grad_y = []
        fy_ = np.zeros_like(M_)
        fy_[:,1:,:] = 2*M_[:,1:,:] - M_[:,:-1,:]
        ind_ = (fy_[:,:,:]==1)
        ind__ = np.zeros_like(ind_)
        ind__[:,:-1,:] = ind_[:,1:,:]
        gi = np.zeros(np.sum(mask),dtype=bool)
        gi[M_ind_[ind_]] = True
        grad_y.append(gi)
        gi = np.zeros(np.sum(mask),dtype=bool)
        gi[M_ind_[ind__]] = True
        grad_y.append(gi)


        # For axe z
        grad_z = []
        fz_ = np.zeros_like(M_)
        fz_[:,:,1:] = 2*M_[:,:,1:] - M_[:,:,:-1]
        ind_ = (fz_[:,:,:]==1)
        ind__ = np.zeros_like(ind_)
        ind__[:,:,:-1] = ind_[:,:,1:]
        gi = np.zeros(np.sum(mask),dtype=bool)
        gi[M_ind_[ind_]] = True
        grad_z.append(gi)
        gi = np.zeros(np.sum(mask),dtype=bool)
        gi[M_ind_[ind__]] = True
        grad_z.append(gi)


        return [grad_x,grad_y,grad_z]


def grad_from_indices(M,grad_ind):
    """
    Compute gradient of an image with indices
    grad is computed as :
    grad = np.zeros_like(X)
    grad[grad_x[1]] = X[grad_x[0]]-X[grad_x[1]]
    """
    grad = np.zeros([len(grad_ind),M.shape[0]])
    # x
    grad[0,grad_ind[0][1]] = M[grad_ind[0][0]] - M[grad_ind[0][1]]
    # y
    grad[1,grad_ind[1][1]] = M[grad_ind[1][0]] - M[grad_ind[1][1]]
    # z
    if len(grad_ind) == 3 :
        grad[2,grad_ind[2][1]] = M[grad_ind[2][0]] - M[grad_ind[2][1]]
    return grad


def indices_for_div(mask):
    """
    Compute indices for the div
    return a list of (div_0,div_1,div_2)
    div is compute as :
    div = np.copy(grad)
    div[div_x[0]] = grad[div_x[0]] - grad[div_x[1]]
    div[div_x[3]] = - grad[div_x[2]]
    """
    ### Construction of the indices matrix
    M_ = mask.astype('int')
    M_ind_ = np.zeros_like(M_)
    M_ind_[mask] = range(np.sum(mask))

    if np.size(mask.shape) == 2 :
            # For axe x
            div_x = []
            fx_ = np.zeros_like(M_)
            fx_[:-1,:] = 2*M_[:-1,:] - M_[1:,:]
            ind_ = (fx_[:,:]==1)
            ind__ = np.zeros_like(ind_)
            ind__[1:,:] = ind_[:-1,:]
            gi = np.zeros(np.sum(mask),dtype=bool)
            gi[M_ind_[ind__]] = True
            div_x.append(gi)
            gi = np.zeros(np.sum(mask),dtype=bool)
            gi[M_ind_[ind_]] = True
            div_x.append(gi)
            border_ = np.zeros_like(mask)
            border_[1:,:] = (fx_[:-1,:]==1)*(fx_[1:,:]==2)
            border_[-1,:] = mask[-1,:]*(fx_[-2,:]==1)
            border__ = np.zeros_like(border_)
            border__[:-1,:] = border_[1:,:]
            gi = np.zeros(np.sum(mask),dtype=bool)
            gi[M_ind_[border__]] = True
            div_x.append(gi)
            gi = np.zeros(np.sum(mask),dtype=bool)
            gi[M_ind_[border_]] = True
            div_x.append(gi)

            # For axe y
            div_y = []
            fy_ = np.zeros_like(M_)
            fy_[:,:-1] = 2*M_[:,:-1] - M_[:,1:]
            ind_ = (fy_[:,:]==1)
            ind__ = np.zeros_like(ind_)
            ind__[:,1:] = ind_[:,:-1]
            gi = np.zeros(np.sum(mask),dtype=bool)
            gi[M_ind_[ind__]] = True
            div_y.append(gi)
            gi = np.zeros(np.sum(mask),dtype=bool)
            gi[M_ind_[ind_]] = True
            div_y.append(gi)
            border_ = np.zeros_like(mask)
            border_[:,1:] = (fy_[:,:-1]==1)*(fy_[:,1:]==2)
            border_[:,-1] = mask[:,-1]*(fy_[:,-2]==1)
            border__ = np.zeros_like(border_)
            border__[:,:-1] = border_[:,1:]
            gi = np.zeros(np.sum(mask),dtype=bool)
            gi[M_ind_[border__]] = True
            div_y.append(gi)
            gi = np.zeros(np.sum(mask),dtype=bool)
            gi[M_ind_[border_]] = True
            div_y.append(gi)

            return [div_x,div_y]

    if np.size(mask.shape) == 3 :
            # For axe x
            div_x = []
            fx_ = np.zeros_like(M_)
            fx_[:-1,:,:] = 2*M_[:-1,:,:] - M_[1:,:,:]
            ind_ = (fx_[:,:,:]==1)
            ind__ = np.zeros_like(ind_)
            ind__[1:,:,:] = ind_[:-1,:,:]
            gi = np.zeros(np.sum(mask),dtype=bool)
            gi[M_ind_[ind__]] = True
            div_x.append(gi)
            gi = np.zeros(np.sum(mask),dtype=bool)
            gi[M_ind_[ind_]] = True
            div_x.append(gi)
            border_ = np.zeros_like(mask)
            border_[1:,:,:] = (fx_[:-1,:,:]==1)*(fx_[1:,:,:]==2)
            border_[-1,:,:] = mask[-1,:,:]*(fx_[-2,:,:]==1)
            border__ = np.zeros_like(border_)
            border__[:-1,:,:] = border_[1:,:,:]
            gi = np.zeros(np.sum(mask),dtype=bool)
            gi[M_ind_[border__]] = True
            div_x.append(gi)
            gi = np.zeros(np.sum(mask),dtype=bool)
            gi[M_ind_[border_]] = True
            div_x.append(gi)

            # For axe y
            div_y = []
            fy_ = np.zeros_like(M_)
            fy_[:,:-1,:] = 2*M_[:,:-1,:] - M_[:,1:,:]
            ind_ = (fy_[:,:,:]==1)
            ind__ = np.zeros_like(ind_)
            ind__[:,1:,:] = ind_[:,:-1,:]
            gi = np.zeros(np.sum(mask),dtype=bool)
            gi[M_ind_[ind__]] = True
            div_y.append(gi)
            gi = np.zeros(np.sum(mask),dtype=bool)
            gi[M_ind_[ind_]] = True
            div_y.append(gi)
            border_ = np.zeros_like(mask)
            border_[:,1:,:] = (fy_[:,:-1,:]==1)*(fy_[:,1:,:]==2)
            border_[:,-1,:] = mask[:,-1,:]*(fy_[:,-2,:]==1)
            border__ = np.zeros_like(border_)
            border__[:,:-1,:] = border_[:,1:,:]
            gi = np.zeros(np.sum(mask),dtype=bool)
            gi[M_ind_[border__]] = True
            div_y.append(gi)
            gi = np.zeros(np.sum(mask),dtype=bool)
            gi[M_ind_[border_]] = True
            div_y.append(gi)

            # For axe z
            div_z = []
            fz_ = np.zeros_like(M_)
            fz_[:,:,:-1] = 2*M_[:,:,:-1] - M_[:,:,1:]
            ind_ = (fz_[:,:,:]==1)
            ind__ = np.zeros_like(ind_)
            ind__[:,:,1:] = ind_[:,:,:-1]
            gi = np.zeros(np.sum(mask),dtype=bool)
            gi[M_ind_[ind__]] = True
            div_z.append(gi)
            gi = np.zeros(np.sum(mask),dtype=bool)
            gi[M_ind_[ind_]] = True
            div_z.append(gi)
            border_ = np.zeros_like(mask)
            border_[:,:,1:] = (fz_[:,:,:-1]==1)*(fz_[:,:,1:]==2)
            border_[:,:,-1] = mask[:,:,-1]*(fz_[:,:,-2]==1)
            border__ = np.zeros_like(border_)
            border__[:,:,:-1] = border_[:,:,1:]
            gi = np.zeros(np.sum(mask),dtype=bool)
            gi[M_ind_[border__]] = True
            div_z.append(gi)
            gi = np.zeros(np.sum(mask),dtype=bool)
            gi[M_ind_[border_]] = True
            div_z.append(gi)

            return [div_x,div_y,div_z]


def div_from_indices(grad,div_ind):
    """
    Compute div of an image with indices
    div is compute as :
    div = np.copy(grad)
    div[div_x[0]] = grad[div_x[0]] - grad[div_x[1]]
    div[div_x[3]] = - grad[div_x[2]]
    """
    # x
    grad_x = grad[0,:]
    div_x = np.copy(grad_x)
    div_x[div_ind[0][0]] -= grad_x[div_ind[0][1]]
    div_x[div_ind[0][3]] = - grad_x[div_ind[0][2]]
    # y
    grad_y = grad[1,:]
    div_y = np.copy(grad_y)
    div_y[div_ind[1][0]] -= grad_y[div_ind[1][1]]
    div_y[div_ind[1][3]] = - grad_y[div_ind[1][2]]
    # z
    if grad.shape[0] == 3:
        grad_z = grad[2,:]
        div_z = np.copy(grad_z)
        div_z[div_ind[2][0]] -= grad_z[div_ind[2][1]]
        div_z[div_ind[2][3]] = - grad_z[div_ind[2][2]]
        return div_x+div_y+div_z
    else:
        return div_x+div_y



