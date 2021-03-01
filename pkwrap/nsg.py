"""implementation of NG-SGD from Kaldi"""
from dataclasses import dataclass
from typing import Sequence
import torch
from math import exp
import logging
import pdb

# keeping this implementation in python for now. even if it is run, I don't
# expect it to be run multiple times.
# TOOD: shift it to C++ biniding
def OrthogonalizeRows(M):
    """Implementation of Gram-Schmidt"""
    num_rows, num_cols = M.shape
    for i in range(num_rows):
        counter = 0
        while True:
            start_prod = M[i, :].pow(2.0).sum()
            if torch.isnan(start_prod) or torch.isinf(start_prod) or start_prod == 0.0:
                M[i, :].randn()
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


@dataclass
class OnlineNaturalGradient:
    """NGState value container"""
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
        self.d_t = None
        self.W_t = None

    def init_orthonormal_special(self):
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
        N, D = X.shape
        R = self.rank
        eta = self._compute_eta(N)
        W_t = self.W_t
        # H_t = X W_t^T
        H_t = X.mm(W_t.T)
        # print("t, X", self.t, X)
        # print("t, W_t", self.t, W_t)
        # print("t, H_t", self.t, H_t)
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
        # print("Z_t ", self.t, Z_t)
        z_t_scale = Z_t.trace().clamp_min(1.0)
        Z_t = Z_t.mul(1.0/z_t_scale)
        # print("Z_t ", self.t, Z_t.min(), Z_t.max(), X.min(), X.max(), K_t.min(), K_t.max(), L_t.min(), L_t.max())
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
        # print("Must reorth after flooring ", self.t, must_reorthogonalize)
        # print("t reorth", self.t, must_reorthogonalize)
        eigvalues.clamp_min_(c_t_floor)
        # print("t, X updated", self.t, X)
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
        threshold = 1.0e-03
        R, D = W_t1.shape
        beta_t1 = OnlineNaturalGradient.get_beta(rho_t1, self.alpha, d_t1, D)
        e_t1, sqrt_e_t1, inv_sqrt_e_t1 = self._compute_et(d_t1, beta_t1)
        # a trick to re-use memory would be to re-use temp_O
        # print("in reorth ", self.t, inv_sqrt_e_t1)
        temp_O.copy_(W_t1.mm(W_t1.T)*inv_sqrt_e_t1[:, None]*inv_sqrt_e_t1[None,:])
        # TODO: check if temp_O is unit matrix
        if _is_unit(temp_O):
            return
        Omat = temp_O.cpu()
        # print("W_t1 ", self.t, W_t1)
        # print("Omat1 ", self.t, Omat)
        cholesky_ok = True
        try:
            Omat_inv = Omat.cholesky().cholesky_inverse()
            # print("Omat inv", self.t, Omat_inv)
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
        D = d_t.shape[0]
        e_t = 1.0/(beta_t/d_t + 1.0)
        sqrt_e_t = e_t.pow(0.5)
        inv_sqrt_e_t = sqrt_e_t.pow(-1.0)
        return e_t, sqrt_e_t, inv_sqrt_e_t

    @staticmethod
    def get_beta(rho_t, alpha, d_t, D):
        return rho_t*(1+alpha) + alpha*d_t.sum()/D

    # keeping this public so that I can test it
    def compute_zt(self, N, inv_sqrt_e_t, K_t, L_t):
        eta = self._compute_eta(N)
        d_t = self.d_t
        rho = self.rho
        d_t_rho_t = d_t + rho
        etaN = eta/N
        eta1 = 1.0-eta
        etaN_sq = etaN*etaN
        eta1_sq = eta1*eta1
        etaN_eta1 = etaN*eta1
        R = d_t.shape[0]
        # so far everything has been in 
        L_t_factor = L_t.cpu().to(torch.double)
        K_t_factor = K_t.cpu().to(torch.double)
        # we need to make sure L_t and K_t are symmetric!
        L_t_factor = L_t_factor + L_t_factor.T
        K_t_factor = K_t_factor + K_t_factor.T
        L_t_factor.mul_(0.5)
        K_t_factor.mul_(0.5)
        inv_sqrt_e_t_cpu = inv_sqrt_e_t.cpu()
        d_t_rho_t_cpu = d_t_rho_t.cpu()
        factor1 = ((inv_sqrt_e_t_cpu*etaN_sq)[:, None] * K_t_factor)*inv_sqrt_e_t_cpu[None,:]
        factor2 = ((inv_sqrt_e_t_cpu*etaN_eta1)[:, None] * L_t_factor)*(inv_sqrt_e_t_cpu*d_t_rho_t_cpu)[None,:]
        factor3 = ((inv_sqrt_e_t_cpu*d_t_rho_t_cpu*etaN_eta1)[:, None] * L_t_factor)*(inv_sqrt_e_t_cpu)[None,:]
        # TODO: factor 2 and factor 3 can be simplied in one expression;        
        # TODO: factor4 can be simplified, but need to check if it is benificial computationally
        factor4 = (eta1_sq*d_t_rho_t_cpu.pow(2.0)).diag()
        Z_t = factor1 + factor2 + factor3 + factor4
        Z_t = (Z_t + Z_t.T).mul(0.5)
        try:
            assert torch.allclose(Z_t, Z_t.T)
        except AssertionError:
            print("Z_t is not symmetric in ", self.t)
            # print("Z_t value is ", Z_t)
            # print("is factor 1 symmetric", torch.allclose(factor1, factor1.T))
            # print("is factor 2 symmetric", torch.allclose(factor2, factor2.T))
            # print("is factor 3 symmetric", torch.allclose(factor3, factor3.T))
            # print("is factor 4 symmetric", torch.allclose(factor4, factor4.T))
            # print("is factor 2+3 symmetric", torch.allclose((factor2+factor3), (factor2.T+factor3.T)))
            # print("is K_t symmetric", torch.allclose(K_t_factor, K_t_factor.T))
            # print("is L_t symmetric", torch.allclose(L_t_factor, L_t_factor.T))

        return Z_t

    @torch.no_grad()
    def precondition_directions(self, X):
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
        self.t += 1

    def validate(self):
        assert self.num_samples_history >0. and self.num_samples_history<=1e+06
        assert self.num_minibatches_history == 0 or self.num_minibatches_history > 1.0
        assert self.num_minibatches_history < 1e+06
        assert self.alpha >= 0.
        assert self.rank > 0
        assert self.epsilon > 0. and self.epsilon <= 1e-05
        assert self.delta > 0. and self.delta <= 1e-02

    # TODO: implement caching
    def _compute_eta(self, N):
        if self.num_minibatches_history > 0.0:
            return 1.0/self.num_minibatches_history
        else:
            # TODO: check if num_samples_history > 0.0
            return min(0.9, 1.0 - exp(-N/self.num_samples_history))

def _is_unit(symmetric_matrix):
    r = symmetric_matrix.shape[0]
    # TODO: much simpler implementation that doesn't require so much memory
    return torch.allclose(symmetric_matrix, torch.eye(r, r, dtype=symmetric_matrix.dtype, device=symmetric_matrix.device))
    # IDEA: check if diag is all 1s then set the diag to 0, temporarily to check if all elements are 0s
    #       the min and max should be close to 0