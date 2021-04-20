"""implementation of NG-SGD from Kaldi

This is an implementation of Natural Gradient Descent based on what is available in Kaldi.
It is reimplemented in pytorch to avoid using Kaldi's GPU libraries when using other toolkits
that require compute-exclusive mode. Most of the implementation is based on Kaldi, while some
vectorization is applied to avoid big for loops.
"""

# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

from dataclasses import dataclass
from typing import Sequence
import torch
from math import exp
import logging

# keeping this implementation in python for now. even if it is run, I don't
# expect it to be run multiple times.
# TOOD: shift it to C++ biniding
def OrthogonalizeRows(M):
    """Implementation of Gram-Schmidt orthognolization

    The input matrix is orthogonalized in place. When nan or inf is produced
    during the computation, the rows are reset with random values. This is done
    at most 100 times at which point an Exception is raised.
    
    Args:
        M: Torch Tensor

    Returns:
        bool: True when successfull, else raises an Exception

    Raises:
        Exception: when looping for more than 100 times.
    """
    num_rows, num_cols = M.shape
    for i in range(num_rows):
        counter = 0
        while True:
            start_prod = M[i, :].pow(2.0).sum()
            if torch.isnan(start_prod) or torch.isinf(start_prod) or start_prod == 0.0:
                M[i, :].normal_()
                counter += 1
                if counter > 100:
                    raise Exception("Loop detected while orthogonalizing matrix")
                continue
            # TODO: vectorize this loop
            for j in range(0, i):
                # this product is useless. why 
                prod = (M[j, :]*M[i,:]).sum()
                M[i, :].add_(M[j, :], alpha=-prod)
            end_prod = M[i, :].pow(2.0).sum()
            if end_prod <= 0.01 * start_prod:
                if end_prod == 0.0:
                    M[i, :].randn()
                counter += 1
                if counter > 100:
                    raise Exception("Loop detected while orthogonalizing matrix")
            else:
                M[i, :].mul_(1.0/end_prod.sqrt())
                break
    return True


@dataclass
class OnlineNaturalGradient:
    """NGState value container
    
    The state values are implemented using dataclass.

    Attributes:
        alpha (float): alpha value of the state (default 4.0)
        num_samples_history: size of history (default 2000.0)
        update_period (int): update W_t every update_period iterations only (default 4)
        num_minibatches_history (int): minibatch history size (default 0)
        epsilon (float): epsilon value used for updates (default 1.0e-10)
        delta (float): delta value used for updates (default 5.0e-4)
        frozen (bool): set when the state should not be updated (default False)
        t (int): number of updates completed so far; starts from 0
        rank (int): rank of the update
        num_initial_updates (int): always update till 'num_initial_updates' (default 10)
    """
    alpha: float = 4.0
    num_samples_history: float = 2000.0
    update_period: int = 4
    num_minibatches_history: int = 0
    epsilon: float = 1.0e-10
    delta: float = 5.0e-4
    frozen: bool = False
    t: int = 0
    rho: float = -1e+10
    rank:int = 40
    num_initial_updates: int = 10

    def __post_init__(self):
        """Set d_t and W_t to None"""
        self.d_t = None
        self.W_t = None

    def init_orthonormal_special(self):
        """initialize value of W_t
        
        Don't call this function directly. It is called by init_default.
        """
        R = self.W_t
        num_rows, num_cols = R.shape
        R.zero_()
        first_elem = 1.1
        first_elem2 = 1.1**2
        for r in range(num_rows):
            cols = [c for c in range(r, num_cols, num_rows)]
            normalizer = 1.0/(first_elem2 + len(cols)-1)**(0.5)
            R[r, cols[0]] = first_elem*normalizer
            R[r, cols[1:]] = normalizer

    def init_default(self, D, device=None):
        """Initialize W_t and d_t
        
        This should be called only once. The rank is set to maximum of D-1.

        Args:
            D: dimension of input data
        """
        if self.rank >= D:
            self.rank = D-1
        if self.rank == 0:
            return
        self.validate()
        self.rho = self.epsilon
        # TODO: decide the device
        if device is None:
            device="cpu"
        self.d_t = torch.zeros(self.rank, device=device).add_(self.epsilon)
        self.W_t = torch.zeros(self.rank, D, device=device)
        self.init_orthonormal_special()
        E_tii = (1.0/(2.0+(D+self.rank)*self.alpha/D))**(0.5)
        self.W_t.mul_(E_tii)
        self.t = 0

    def init(self, X):
        """Iniitalize the values of W_t and d_t.

        This function calls init_default. We may just merge them later.
        """
        D = X.shape[-1]
        self.init_default(D, device=X.device)
        self.t = 1
        num_init_iters = 3
        if X.shape[0] <= self.rank:
            num_init_iters = 1
        for _ in range(num_init_iters):
            Xcopy = torch.zeros_like(X, requires_grad=False).copy_(X)
            self.precondition_directions(Xcopy)

    @torch.no_grad()
    def _precondition_directions_internal(self, X, initial_product):
        """Internal function to precondition the input and gradients"""
        N, D = X.shape
        R = self.rank
        eta = self._compute_eta(N)
        W_t = self.W_t
        # H_t = X W_t^T
        H_t = X.mm(W_t.T)
        if self.t > self.num_initial_updates and (self.t-self.num_initial_updates) % self.update_period != 0:
            # X <- X - H_t W_t
            X.add_(H_t.mm(W_t), alpha=-1.0)
            return 
        J_t = H_t.T.mm(X)
        # TODO: compute LK together because in GPUs that would mean only one call
        L_t = W_t.mm(J_t.T)
        K_t = J_t.mm(J_t.T)

        alpha = self.alpha
        d_t = self.d_t
        rho_t = self.rho
        beta_t = OnlineNaturalGradient.get_beta(rho_t, alpha, d_t, D)
        inv_sqrt_e_t = self._compute_et(d_t, beta_t)[-1]
        # TODO: check if doing this on CPU is faster. Kaldi does that
        Z_t = self.compute_zt(N, inv_sqrt_e_t, K_t, L_t)
        z_t_scale = Z_t.trace().clamp_min(1.0)
        Z_t = Z_t.mul(1.0/z_t_scale)
        Z_t = Z_t.to(dtype=torch.float)
        eigvalues, U = Z_t.eig(eigenvectors=True)
        eigvalues_sorted = eigvalues[:,0].sort(descending=True)
        # TODO: remove sorting. not really required
        eigvalues = eigvalues_sorted.values
        U = U[:, eigvalues_sorted.indices].cuda()
        eigvalues.mul_(z_t_scale)

        condition_threshold = 1.0e+06
        must_reorthogonalize = eigvalues.max() > condition_threshold*eigvalues.min()
        c_t_floor = torch.tensor((rho_t*(1-eta))**2, device=eigvalues.device, requires_grad=False)
        if any(eigvalues<c_t_floor):
            must_reorthogonalize = True
        eigvalues.clamp_min_(c_t_floor)
        sqrt_c_t = eigvalues.pow(0.5).cuda()
        rho_t1 = (1.0)/(D-R)*(eta/N*initial_product + (1-eta)*(D*rho_t+d_t.sum()) - sqrt_c_t.sum())
        d_t1 = sqrt_c_t - rho_t1
        floor_val = torch.max(torch.tensor((self.delta*sqrt_c_t.max(), self.epsilon)))
        if rho_t1 < floor_val:
            rho_t1 = floor_val
        d_t1.clamp_min_(floor_val)

        X.add_(H_t.mm(W_t), alpha=-1.0)
        W_t1 = self._compute_Wt1(N, d_t1, rho_t1, U, sqrt_c_t, inv_sqrt_e_t, J_t)
        if must_reorthogonalize:
            self._reorthogonalize_Rt1(d_t1, rho_t1, W_t1, L_t)
        self.W_t.copy_(W_t1.to(self.W_t.device))
        self.d_t.copy_(d_t1.to(self.d_t.device))
        self.rho = rho_t1
        return

    def _compute_Wt1(self, N, d_t1, rho_t1, U, sqrt_c_t, inv_sqrt_et, J_t):
        """internal function to recompute W_t for next t"""
        d_t = self.d_t
        rho_t = self.rho
        W_t = self.W_t
        # TOOD: do we really need to create another copy?
        R, D = W_t.shape
        eta = self._compute_eta(N)
        beta_t1 = OnlineNaturalGradient.get_beta(rho_t1, self.alpha, d_t1, D)
        sqrt_e_t1 = self._compute_et(d_t1, beta_t1)[1]
        inv_sqrt_c_t = sqrt_c_t.pow(-1.0)
        w_t_coeff = ((1.0-eta)/(eta/N) * (d_t+rho_t)).cuda()
        # this is in CPU
        A_t = U.T * ((eta/N)*sqrt_e_t1*inv_sqrt_c_t)[:, None] * inv_sqrt_et[None, :]
        A_t = A_t.cuda()
        J_t.add_(w_t_coeff[:, None]*W_t)
        W_t1 = A_t.mm(J_t)
        # print("W_t1 range", W_t1.min(), W_t1.max(), J_t.min(), J_t.max(), A_t.min(), A_t.max())
        return W_t1

    def _reorthogonalize_Rt1(self, d_t1, rho_t1, W_t1, temp_O):
        """recompute R_t for next t"""
        threshold = 1.0e-03
        R, D = W_t1.shape
        beta_t1 = OnlineNaturalGradient.get_beta(rho_t1, self.alpha, d_t1, D)
        e_t1, sqrt_e_t1, inv_sqrt_e_t1 = self._compute_et(d_t1, beta_t1)
        # a trick to re-use memory would be to re-use temp_O
        temp_O.copy_(W_t1.mm(W_t1.T)*inv_sqrt_e_t1[:, None]*inv_sqrt_e_t1[None,:])
        # TODO: check if temp_O is unit matrix
        if _is_unit(temp_O):
            return
        Omat = temp_O.cpu()
        cholesky_ok = True
        try:
            Omat_inv = Omat.cholesky().cholesky_inverse()
            if Omat_inv.max() > 100.:
                logging.warning("Cholesky out of range. Using Gram-Schmidt t={} {}".format(self.t, Omat_inv.max()))
                raise Exception("Cholesky out of range. Using Gram-Schmidt")
            Omat_inv = Omat_inv.cuda()
            Omat_inv.mul_(sqrt_e_t1[:, None]).mul_(inv_sqrt_e_t1[None, :])
            # TODO: check if we reallyt need this copy_. I don't think temp_O is used anymore
            temp_O.copy_(Omat_inv)
            W_t1.copy_(temp_O.mm(W_t1))
            return
        except:
            # must reorth with Gram-Schmidt
            cholesky_ok = False
        if not cholesky_ok:
            logging.info("Running gram schmidt t={}".format(self.t))
            W_t1_cpu = W_t1.cpu()
            OrthogonalizeRows(W_t1_cpu)
            W_t1.copy_(W_t1_cpu.cuda())
            W_t1.mul_(sqrt_e_t1.cuda()[:, None])
            return

    def _compute_et(self, d_t, beta_t):
        """return e_t, sqrt_e_t and inv_sqrt_e_t given d_t and beta_t"""
        D = d_t.shape[0]
        e_t = 1.0/(beta_t/d_t + 1.0)
        sqrt_e_t = e_t.pow(0.5)
        inv_sqrt_e_t = sqrt_e_t.pow(-1.0)
        return e_t, sqrt_e_t, inv_sqrt_e_t

    @staticmethod
    def get_beta(rho_t, alpha, d_t, D):
        """beta = rho_t*(1+alpha) + alpha*d_t.sum()/D"""
        return rho_t*(1+alpha) + alpha*d_t.sum()/D

    def compute_zt(self, N, inv_sqrt_e_t, K_t, L_t):
        """return new value of Z_t"""
        eta, d_t, rho = self._compute_eta(N), self.d_t, self.rho
        R = d_t.shape[0]
        d_t_rho_t = d_t + rho
        etaN, eta1 = (eta/N, 1.0-eta)
        etaN_sq, eta1_sq, etaN_eta1 = etaN*etaN, eta1*eta1, etaN*eta1
        # so far everything has been in the device of the input.
        L_t_factor = L_t.cpu().to(torch.double)
        K_t_factor = K_t.cpu().to(torch.double)
        # we need to make sure L_t and K_t are symmetric!
        L_t_factor = L_t_factor + L_t_factor.T
        K_t_factor = K_t_factor + K_t_factor.T
        L_t_factor.mul_(0.5)
        K_t_factor.mul_(0.5)
        # make sure other necessary variables are in CPU
        inv_sqrt_e_t_cpu, d_t_rho_t_cpu = inv_sqrt_e_t.cpu(), d_t_rho_t.cpu()
        # there are four factors in the original code. I split them here so that it is
        # easier to see what's going on.
        factor1 = ((inv_sqrt_e_t_cpu*etaN_sq)[:, None] * K_t_factor)*inv_sqrt_e_t_cpu[None,:]
        factor2 = ((inv_sqrt_e_t_cpu*etaN_eta1)[:, None] * L_t_factor)*(inv_sqrt_e_t_cpu*d_t_rho_t_cpu)[None,:]
        factor3 = ((inv_sqrt_e_t_cpu*d_t_rho_t_cpu*etaN_eta1)[:, None] * L_t_factor)*(inv_sqrt_e_t_cpu)[None,:]
        # TODO: factor 2 and factor 3 can be simplied in one expression;        
        # TODO: factor4 can be simplified, but need to check if it is benificial computationally
        factor4 = (eta1_sq*d_t_rho_t_cpu.pow(2.0)).diag()
        Z_t = factor1 + factor2 + factor3 + factor4
        # TODO: avoid this by making sure factor2+3 is symmetric
        Z_t = (Z_t + Z_t.T).mul(0.5)
        return Z_t

    @torch.no_grad()
    def precondition_directions(self, X):
        """Runs one step of preconditioning on X
        
        Args:
            X: a two-dimensional matrix, usually input to the layer or grad_out

        Returns:
            None
        """
        if self.t == 0:
            self.init(X)
            self.t = 0
        initial_product = X.pow(2.0).sum()
        self._precondition_directions_internal(X,  initial_product)
        if initial_product <= 0.0:
            scale = 1.
        else:
            final_product = X.pow(2.0).sum()
            # print("ip fp ", initial_product, final_product)
            scale = (initial_product/final_product).pow(0.5)
        self.step()
        return scale

    def step(self):
        """Increment t by 1"""
        self.t += 1

    def validate(self):
        """Method to check if the state values are sane"""
        assert self.num_samples_history >0. and self.num_samples_history<=1e+06
        assert self.num_minibatches_history == 0 or self.num_minibatches_history > 1.0
        assert self.num_minibatches_history < 1e+06
        assert self.alpha >= 0.
        assert self.rank > 0
        assert self.epsilon > 0. and self.epsilon <= 1e-05
        assert self.delta > 0. and self.delta <= 1e-02

    # TODO: implement caching
    def _compute_eta(self, N):
        """return eta value given N"""
        if self.num_minibatches_history > 0.0:
            return 1.0/self.num_minibatches_history
        else:
            # TODO: check if num_samples_history > 0.0
            return min(0.9, 1.0 - exp(-N/self.num_samples_history))

def _is_unit(symmetric_matrix):
    """Check if a matrix is a Unit matrix

    This is a private method for the moment.

    Args:
        symmetric_matrix: input is assumed to be a symmetric matrix

    Returns:
        bool: True if the matrix is Unit, False otherwise
    """
    r = symmetric_matrix.shape[0]
    # TODO: much simpler implementation that doesn't require so much memory
    return torch.allclose(symmetric_matrix, torch.eye(r, r, dtype=symmetric_matrix.dtype, device=symmetric_matrix.device))
    # IDEA: check if diag is all 1s then set the diag to 0, temporarily to check if all elements are 0s
    #       the min and max should be close to 0
