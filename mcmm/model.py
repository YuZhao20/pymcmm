# -*- coding: utf-8 -*-
"""
Mixed-Copula Mixture Model (MCMM) with Gaussian Copula

This library provides the core implementation of the MCMM model,
capable of clustering datasets with mixed continuous, categorical,
and ordinal data types.
"""

import numpy as np
import pandas as pd
from numpy.linalg import eigh
from scipy.stats import norm, t as student_t
from scipy.special import logsumexp
from scipy.optimize import minimize, minimize_scalar
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import List, Optional, Dict, Tuple
from pandas.api.types import CategoricalDtype
from joblib import Parallel, delayed

# ---------- Utility Functions ----------

def _safe_log(x, eps=1e-12):
    """Computes log safely, avoiding log(0)."""
    return np.log(np.clip(x, eps, None))

def _nearest_pd(A, eps=1e-7):
    """Finds the nearest positive definite matrix to A."""
    A = 0.5 * (A + A.T)
    vals, vecs = eigh(A)
    vals = np.clip(vals, eps, None)
    return (vecs * vals) @ vecs.T

def _shrink_corr(R, lam=0.05):
    """Applies shrinkage to a correlation matrix."""
    d = R.shape[0]
    S = (1.0 - lam) * R + lam * np.eye(d)
    return _nearest_pd(0.5 * (S + S.T))

def _submatrix(M, idx):
    """Extracts a submatrix from M based on indices."""
    idx = np.asarray(idx, dtype=int)
    if M.ndim == 2 and len(idx) > 0:
        return M[np.ix_(idx, idx)]
    return M

def _weighted_onehot_counts(values, categories, weights):
    """Computes weighted counts for categorical data."""
    mask = pd.isna(values)
    vals, w = values[~mask], weights[~mask]
    idx = pd.Categorical(vals, categories=categories, ordered=False).codes
    L = len(categories)
    if len(w) > 0:
        valid_idx = idx != -1
        return np.bincount(idx[valid_idx], weights=w[valid_idx], minlength=L)
    return np.zeros(L, float)

# ---------- Continuous Marginal Distributions ----------

def _gaussian_logdensity_scalar(x, mu, sig):
    z = (x - mu) / np.clip(sig, 1e-9, None)
    return -0.5 * z * z - _safe_log(sig) - 0.5 * np.log(2 * np.pi)

def _gaussian_cdf_scalar(x, mu, sig):
    return norm.cdf((x - mu) / np.clip(sig, 1e-9, None))

def _studentt_logdensity_scalar(x, mu, sig, nu):
    z = (x - mu) / np.clip(sig, 1e-9, None)
    return student_t.logpdf(z, df=nu) - _safe_log(sig)

def _studentt_cdf_scalar(x, mu, sig, nu):
    z = (x - mu) / np.clip(sig, 1e-9, None)
    return student_t.cdf(z, df=nu)

# ---------- Copula Densities ----------

def _log_gaussian_copula_density_full(u, R):
    m = len(u)
    if m == 0: return 0.0
    z = norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
    try:
        sign, logdet = np.linalg.slogdet(R)
        if sign <= 0:
            R = _nearest_pd(R, 1e-7)
            sign, logdet = np.linalg.slogdet(R)
        invR = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        R = _nearest_pd(R, 1e-7)
        sign, logdet = np.linalg.slogdet(R)
        invR = np.linalg.inv(R)
    quad = z @ (invR - np.eye(m)) @ z
    return -0.5 * logdet - 0.5 * quad

def _log_bivariate_gaussian_copula(u1, u2, rho):
    u1, u2 = np.clip(u1, 1e-10, 1 - 1e-10), np.clip(u2, 1e-10, 1 - 1e-10)
    z1, z2 = norm.ppf(u1), norm.ppf(u2)
    r2 = rho * rho
    if 1 - r2 < 1e-9:
        return -np.inf if (z1 - np.sign(rho) * z2)**2 > 1e-3 else 0
    
    log_phi2 = -0.5 * np.log(1 - r2) - (z1*z1 - 2*rho*z1*z2 + z2*z2) / (2 * (1 - r2))
    log_phi_sum = -0.5 * (z1**2 + z2**2)
    return log_phi2 - log_phi_sum

# ---------- Ordinal Model Fitting ----------

def _cumlogit_negloglik(params_raw, counts):
    L = len(counts)
    if L <= 1: return 0.0
    thetas = np.zeros(L - 1)
    thetas[0] = params_raw[0]
    if L > 2: thetas[1:] = params_raw[0] + np.cumsum(np.exp(params_raw[1:]))
    sig = 1 / (1 + np.exp(-thetas))
    probs = np.zeros(L)
    probs[0] = sig[0]
    if L > 2: probs[1:L-1] = np.diff(sig)
    if L >= 2: probs[-1] = 1 - sig[-1]
    return -np.sum(counts * _safe_log(probs))

def _fit_cumlogit_weighted(counts):
    L = len(counts)
    if L <= 1 or counts.sum() < 1e-9:
        return np.array([]), np.ones(L) / max(L, 1)
    cum_probs = np.clip(np.cumsum(counts) / counts.sum(), 1e-6, 1 - 1e-6)
    initial_thetas = -np.log(1 / cum_probs[:-1] - 1)
    x0 = np.zeros(L - 1)
    x0[0] = initial_thetas[0]
    if L > 2: x0[1:] = np.log(np.clip(np.diff(initial_thetas), 1e-6, None))
    res = minimize(_cumlogit_negloglik, x0=x0, args=(counts,), method='L-BFGS-B')
    params = res.x
    thetas = np.zeros(L - 1)
    thetas[0] = params[0]
    if L > 2: thetas[1:] = params[0] + np.cumsum(np.exp(params[1:]))
    sig = 1 / (1 + np.exp(-thetas))
    probs = np.zeros(L)
    probs[0] = sig[0]
    if L > 2: probs[1:L-1] = np.diff(sig)
    if L >= 2: probs[-1] = 1 - sig[-1]
    probs = np.clip(probs, 1e-12, 1.0)
    return thetas, probs / probs.sum()

# ---------- nu Optimizer for Student-t ----------
def _optimize_t_nu(z_list, w_list, nu_bounds=(2.1, 100.0)):
    if not z_list: return None
    z_flat, w_flat = np.concatenate(z_list), np.concatenate(w_list)
    def neg_obj(nu):
        return -np.sum(w_flat * student_t.logpdf(z_flat, df=nu))
    res = minimize_scalar(neg_obj, bounds=nu_bounds, method='bounded', options={'xatol': 1e-2})
    return float(np.clip(res.x, nu_bounds[0], nu_bounds[1]))

# ---------- Speedy Mode Helpers ----------
def _max_spanning_tree_from_corr(Rabs: np.ndarray):
    d = Rabs.shape[0]
    if d == 0: return []
    in_tree = np.zeros(d, dtype=bool)
    in_tree[0] = True
    edges = []
    for _ in range(d - 1):
        best_u, best_v, best_w = -1, -1, -1.0
        idx_u = np.where(in_tree)[0]
        idx_v = np.where(~in_tree)[0]
        if len(idx_v) == 0: break
        for u in idx_u:
            w = Rabs[u, idx_v]
            j = np.argmax(w)
            if w[j] > best_w:
                best_w, best_u, best_v = float(w[j]), u, int(idx_v[j])
        if best_v == -1: break
        in_tree[best_v] = True
        edges.append((best_u, best_v))
    return edges

def _knn_graph_edges(Rabs: np.ndarray, k_per_node: int = 3):
    d = Rabs.shape[0]
    E = set()
    for i in range(d):
        cand = np.argsort(-Rabs[i])
        picked = 0
        for j in cand:
            if i == j: continue
            E.add(tuple(sorted((i, j))))
            picked += 1
            if picked >= k_per_node: break
    return list(E)

# ---------- Main Model Class ----------

class MCMMGaussianCopula:
    """
    Mixed-Copula Mixture Model (MCMM) with Gaussian Copula.
    """
    def __init__(self, n_components: int = 3, max_iter: int = 100, tol: float = 1e-4,
                 cont_marginal: str = 'student_t', t_nu: float = 5.0, estimate_nu: bool = True,
                 ord_marginal: str = 'cumlogit', copula_likelihood: str = 'full',
                 pairwise_weight: str = 'abs_rho', dt_mode: str = 'mid',
                 shrink_lambda: float = 0.05, random_state: Optional[int] = None,
                 verbose: int = 0, n_jobs: int = 1):
        
        # Parameter validation
        assert cont_marginal in ('gaussian', 'student_t')
        assert ord_marginal in ('freq', 'cumlogit')
        assert copula_likelihood in ('full', 'pairwise')
        assert pairwise_weight in ('uniform', 'abs_rho')
        assert dt_mode in ('mid', 'random')

        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.cont_marginal = cont_marginal
        self.t_nu = float(t_nu)
        self.estimate_nu = estimate_nu
        self.ord_marginal = ord_marginal
        self.copula_likelihood = copula_likelihood
        self.pairwise_weight = pairwise_weight
        self.dt_mode = dt_mode
        self.shrink_lambda = shrink_lambda
        self.random_state = np.random.default_rng(random_state)
        self.verbose = verbose
        self.n_jobs = n_jobs
        self._reset_fitted_attributes()

    def _reset_fitted_attributes(self):
        """Initializes or resets attributes that are set during fitting."""
        self.cont_cols_, self.cat_cols_, self.ord_cols_ = None, None, None
        self.cat_levels_, self.ord_levels_ = {}, {}
        self.mu_, self.sig_, self.R_, self.pi_ = None, None, None, None
        self.cat_probs_, self.ord_probs_, self.ord_thetas_ = {}, {}, {}
        self.bic_, self.cbic_, self.loglik_, self.history_, self.fitted_nu_ = None, None, None, [], self.t_nu

    def _infer_columns(self, df, cont_cols, cat_cols, ord_cols):
        if cont_cols is None:
            cont_cols = df.select_dtypes(include=np.number).columns.tolist()
        if cat_cols is None and ord_cols is None:
            cat_like_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
            ord_cols = [c for c in cat_like_cols if isinstance(df[c].dtype, CategoricalDtype) and df[c].cat.ordered]
            cat_cols = [c for c in cat_like_cols if c not in ord_cols]
        return cont_cols or [], cat_cols or [], ord_cols or []
        
    def _prepare_levels(self, df):
        for c in self.cat_cols_:
            self.cat_levels_[c] = sorted([str(x) for x in df[c].dropna().unique()])
        for c in self.ord_cols_:
            if isinstance(df[c].dtype, CategoricalDtype) and df[c].cat.ordered:
                self.ord_levels_[c] = list(df[c].cat.categories)
            else:
                self.ord_levels_[c] = sorted(pd.Series(df[c].dropna().unique()))

    def _init_marginals(self, df):
        n_cont = len(self.cont_cols_)
        self.mu_ = np.zeros((self.K, n_cont)); self.sig_ = np.ones((self.K, n_cont))
        if n_cont > 0:
            Xc = df[self.cont_cols_].to_numpy(float)
            mu0, sig0 = np.nanmean(Xc, axis=0), np.nanstd(Xc, axis=0, ddof=1)
            sig0 = np.where(sig0 < 1e-6, 1.0, sig0)
            for k in range(self.K):
                self.mu_[k] = self.random_state.normal(mu0, sig0 * 0.5)
                self.sig_[k] = sig0

        for c in self.cat_cols_:
            levels = self.cat_levels_[c]
            self.cat_probs_[c] = self.random_state.dirichlet(np.ones(len(levels)), size=self.K)

        for c in self.ord_cols_:
            levels = self.ord_levels_[c]
            self.ord_probs_[c] = self.random_state.dirichlet(np.ones(len(levels)), size=self.K)
            self.ord_thetas_[c] = np.zeros((self.K, max(1, len(levels) - 1)))
        
        d_all = len(self.cont_cols_) + len(self.cat_cols_) + len(self.ord_cols_)
        self.R_ = np.array([np.eye(d_all) for _ in range(self.K)])
        self.pi_ = np.ones(self.K) / self.K
        self.fitted_nu_ = self.t_nu

    def _init_resp_kmeans(self, df):
        df_imputed = df.copy()
        for col in df_imputed.columns:
            if pd.api.types.is_numeric_dtype(df_imputed[col]):
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())
            else:
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode()[0])
        
        xs = []
        if self.cont_cols_:
            xs.append(StandardScaler().fit_transform(df_imputed[self.cont_cols_]))
        if self.cat_cols_ or self.ord_cols_:
            xs.append(OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit_transform(df_imputed[self.cat_cols_ + self.ord_cols_]))
        
        if not xs: return np.ones((len(df), self.K)) / self.K
        
        Xkm = np.hstack(xs)
        kmeans = KMeans(self.K, n_init='auto', random_state=self.random_state.integers(1e6))
        labels = kmeans.fit_predict(Xkm)
        resp = np.zeros((len(df), self.K))
        resp[np.arange(len(df)), labels] = 1.0
        return resp * 0.9 + 0.1 / self.K

    def _build_u_vector(self, row_tuple, k):
        u_list, log_marg = [], 0.0
        
        for j, col in enumerate(self.cont_cols_):
            x = getattr(row_tuple, col)
            if pd.isna(x): u_list.append(np.nan); continue
            mu, sig = self.mu_[k, j], self.sig_[k, j]
            if self.cont_marginal == 'gaussian':
                log_marg += _gaussian_logdensity_scalar(x, mu, sig)
                u_list.append(_gaussian_cdf_scalar(x, mu, sig))
            else:
                log_marg += _studentt_logdensity_scalar(x, mu, sig, self.fitted_nu_)
                u_list.append(_studentt_cdf_scalar(x, mu, sig, self.fitted_nu_))

        for c in self.cat_cols_ + self.ord_cols_:
            x = getattr(row_tuple, c)
            if pd.isna(x): u_list.append(np.nan); continue
            levels = self.cat_levels_.get(c) or self.ord_levels_.get(c)
            probs = self.cat_probs_.get(c, self.ord_probs_.get(c))[k]
            try:
                x_str = str(x)
                idx = levels.index(x_str if c in self.cat_levels_ else x)
                log_marg += _safe_log(probs[idx])
                Fm, F = (np.sum(probs[:idx]), np.sum(probs[:idx+1]))
            except (ValueError, IndexError):
                log_marg += -30; Fm, F = 0.0, 1.0
            u = 0.5 * (Fm + F) if self.dt_mode == 'mid' else self.random_state.uniform(Fm, F)
            u_list.append(u)
            
        return np.array(u_list, float), log_marg
        
    def _log_pk_row(self, row_tuple):
        log_pk = np.zeros(self.K)
        all_cols_map = {c: i for i, c in enumerate(self.cont_cols_ + self.cat_cols_ + self.ord_cols_)}
        obs_idx = [all_cols_map[c] for c in row_tuple._fields if not pd.isna(getattr(row_tuple, c))]
        
        for k in range(self.K):
            u, log_marg = self._build_u_vector(row_tuple, k)
            if not obs_idx: continue
            
            sub_u = u[obs_idx]
            if np.any(np.isnan(sub_u)): log_c = 0.0
            elif self.copula_likelihood == 'full':
                log_c = _log_gaussian_copula_density_full(sub_u, _submatrix(self.R_[k], obs_idx))
            else: # pairwise
                m = len(sub_u)
                if m <= 1: log_c = 0.0
                else:
                    Rsub = _submatrix(self.R_[k], obs_idx)
                    s, wsum = 0.0, 0.0
                    for i in range(m):
                        for j in range(i + 1, m):
                            rho = Rsub[i, j]
                            w = 1.0 if self.pairwise_weight == 'uniform' else abs(rho)
                            s += w * _log_bivariate_gaussian_copula(sub_u[i], sub_u[j], rho)
                            wsum += w
                    log_c = s / max(wsum, 1e-9)
            log_pk[k] = log_marg + log_c
        return log_pk

    def _E_step(self, df):
        row_tuples = list(df.itertuples(index=False, name='DataRow'))
        log_resp_list = Parallel(n_jobs=self.n_jobs)(delayed(self._log_pk_row)(rt) for rt in row_tuples)
        log_resp = np.array(log_resp_list)

        log_resp += _safe_log(self.pi_)
        ll_per_sample = logsumexp(log_resp, axis=1)
        loglik = np.sum(ll_per_sample)
        log_resp -= ll_per_sample[:, None]
        return np.exp(log_resp), loglik

    def _M_step_marginals(self, df, resp):
        """M-step for marginal distribution parameters."""
        if self.cont_cols_:
            Xc = df[self.cont_cols_].to_numpy(float)
            all_z_t, all_w_t = [], []
            
            for k in range(self.K):
                w_em = resp[:, k]
                for j, col in enumerate(self.cont_cols_):
                    x_obs = Xc[:, j]; mask = ~np.isnan(x_obs)
                    if not mask.any(): continue
                    
                    weights = w_em[mask].copy()
                    x_masked = x_obs[mask]

                    if self.cont_marginal == 'student_t':
                        mu_old, sig_old = self.mu_[k, j], self.sig_[k, j]
                        z = (x_masked - mu_old) / np.clip(sig_old, 1e-9, None)
                        w_t = (self.fitted_nu_ + 1) / (self.fitted_nu_ + z**2)
                        weights *= w_t
                        if self.estimate_nu:
                           all_z_t.append(z); all_w_t.append(weights)

                    w_sum = weights.sum()
                    if w_sum < 1e-9: continue
                    
                    mu = np.sum(weights * x_masked) / w_sum
                    var = np.sum(weights * (x_masked - mu)**2) / w_sum
                    self.mu_[k, j], self.sig_[k, j] = mu, np.sqrt(var)

        for c in self.cat_cols_:
            levels, x = self.cat_levels_[c], df[c].to_numpy(dtype=str)
            probs = np.vstack([_weighted_onehot_counts(x, levels, resp[:, k]) for k in range(self.K)])
            self.cat_probs_[c] = (probs + 1e-9) / (probs.sum(axis=1, keepdims=True) + len(levels) * 1e-9)

        for c in self.ord_cols_:
            levels, x = self.ord_levels_[c], df[c].to_numpy()
            for k in range(self.K):
                cnt = _weighted_onehot_counts(x, levels, resp[:, k])
                if self.ord_marginal == 'cumlogit':
                    thetas, probs = _fit_cumlogit_weighted(cnt)
                    if len(thetas) > 0: self.ord_thetas_[c][k, :len(thetas)] = thetas
                    self.ord_probs_[c][k, :] = probs
                else: # 'freq'
                    self.ord_probs_[c][k, :] = (cnt + 1e-9) / (cnt.sum() + len(levels) * 1e-9)

        self.pi_ = resp.mean(axis=0)

        if self.cont_marginal == 'student_t' and self.estimate_nu and all_z_t:
            self.fitted_nu_ = _optimize_t_nu(all_z_t, all_w_t) or self.fitted_nu_

    def _pairwise_weighted_corr(self, Z, W):
        _, d = Z.shape
        R = np.eye(d)
        for i in range(d):
            for j in range(i+1, d):
                mask = ~np.isnan(Z[:, i]) & ~np.isnan(Z[:, j])
                if not np.any(mask): R[i,j]=R[j,i]=0.0; continue
                w_sub, z_i, z_j = W[mask], Z[mask, i], Z[mask, j]
                w_sum = w_sub.sum()
                if w_sum < 1e-9: R[i,j]=R[j,i]=0.0; continue
                mu_i, mu_j = np.sum(w_sub*z_i)/w_sum, np.sum(w_sub*z_j)/w_sum
                cov = np.sum(w_sub*(z_i-mu_i)*(z_j-mu_j))/w_sum
                var_i, var_j = np.sum(w_sub*(z_i-mu_i)**2)/w_sum, np.sum(w_sub*(z_j-mu_j)**2)/w_sum
                rho = cov/np.sqrt(var_i*var_j) if var_i>1e-9 and var_j>1e-9 else 0.0
                R[i,j] = R[j,i] = np.clip(rho, -0.999, 0.999)
        return R

    def _M_step_copulas(self, df, resp):
        """M-step for copula parameters (correlation matrices)."""
        d_all = len(self.cont_cols_) + len(self.cat_cols_) + len(self.ord_cols_)
        U = np.zeros((len(df), self.K, d_all))
        row_tuples = list(df.itertuples(index=False, name='DataRow'))
        for i, rt in enumerate(row_tuples):
            for k in range(self.K):
                U[i, k, :], _ = self._build_u_vector(rt, k)
            
        for k in range(self.K):
            Z = norm.ppf(np.clip(U[:, k, :], 1e-10, 1 - 1e-10))
            R = self._pairwise_weighted_corr(Z, resp[:, k])
            R = _shrink_corr(R, self.shrink_lambda) if self.shrink_lambda > 0 else _nearest_pd(R)
            
            diag = np.sqrt(np.diag(R))
            if np.any(diag < 1e-9): continue
            R /= np.outer(diag, diag)
            np.fill_diagonal(R, 1.0)
            self.R_[k] = R

    def _M_step(self, df, resp):
        """Combined M-step."""
        self._M_step_marginals(df, resp)
        self._M_step_copulas(df, resp)

    def fit(self, df: pd.DataFrame, cont_cols=None, cat_cols=None, ord_cols=None):
        self._reset_fitted_attributes()
        df = df.copy()
        
        self.cont_cols_, self.cat_cols_, self.ord_cols_ = self._infer_columns(df, cont_cols, cat_cols, ord_cols)
        self._prepare_levels(df)
        self._init_marginals(df)
        resp = self._init_resp_kmeans(df)

        prev_ll = -np.inf
        
        for it in range(1, self.max_iter + 1):
            self._M_step(df, resp)
            resp, ll = self._E_step(df)
            self.history_.append(ll)
            
            if self.verbose and (it == 1 or it % 5 == 0):
                print(f"[EM] iter={it:03d}  loglik={ll:.3f}  nu={self.fitted_nu_:.2f}")
            
            if abs(ll - prev_ll) < self.tol * (1.0 + abs(prev_ll)):
                if self.verbose: print(f"Converged at iter {it}, loglik={ll:.3f}")
                break
            prev_ll = ll

        self.loglik_ = prev_ll
        
        d_all = len(self.cont_cols_) + len(self.cat_cols_) + len(self.ord_cols_)
        n_params = (self.K - 1) + self.K * (2*len(self.cont_cols_) + d_all*(d_all-1)//2)
        for c in self.cat_cols_: n_params += self.K * (len(self.cat_levels_[c]) - 1)
        for c in self.ord_cols_: n_params += self.K * (len(self.ord_levels_[c]) - 1)
        if self.cont_marginal == 'student_t' and self.estimate_nu: n_params += 1
        
        self.bic_ = -2 * self.loglik_ + n_params * np.log(len(df))
        # cBIC is only relevant for composite likelihood
        if self.copula_likelihood == 'pairwise':
             self.cbic_ = self._calculate_cbic(df, n_params)
        return self

    def _calculate_cbic(self, df, n_params_bic):
        # This is a placeholder for the actual cBIC calculation which requires
        # estimating the Godambe Information Matrix. For simplicity, we can
        # return a modified BIC or NaN if not fully implemented.
        # A full implementation would require computing scores and hessians.
        # For now, we return NaN as a clear indicator.
        return np.nan

    def predict(self, df):
        return self.predict_proba(df).argmax(axis=1)

    def predict_proba(self, df):
        resp, _ = self._E_step(df.copy())
        return resp
        
    def score_samples(self, df):
        row_tuples = list(df.itertuples(index=False, name='DataRow'))
        log_resp = Parallel(n_jobs=self.n_jobs)(delayed(self._log_pk_row)(rt) for rt in row_tuples)
        log_resp = np.array(log_resp)
        log_resp += _safe_log(self.pi_)
        return logsumexp(log_resp, axis=1)

    def detect_outliers(self, df: pd.DataFrame, q: float = 1.0):
        logs = self.score_samples(df)
        thr = np.percentile(logs, q)
        return logs < thr, logs, thr

# ---------- Speedy Mode Class Definition ----------

class MCMMGaussianCopulaSpeedy(MCMMGaussianCopula):
    """
    Speed-oriented variant of MCMM.
    """
    def __init__(self, *args,
                 speedy_graph:str='mst',
                 speedy_k_per_node:int=3,
                 corr_subsample:int=3000,
                 e_step_batch:int=4096,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.speedy_graph = speedy_graph
        self.speedy_k_per_node = int(speedy_k_per_node)
        self.corr_subsample = int(corr_subsample)
        self.e_step_batch = int(e_step_batch)
        self.speedy_edges_ = None

    def _M_step_copulas(self, df, resp):
        """Overridden M-step for copulas using subsampling and sparse graph."""
        n, d_all = len(df), len(self.cont_cols_) + len(self.cat_cols_) + len(self.ord_cols_)
        self.R_ = np.array([np.eye(d_all) for _ in range(self.K)])
        self.speedy_edges_ = [[] for _ in range(self.K)]

        U_all = np.zeros((n, self.K, d_all))
        row_tuples = list(df.itertuples(index=False, name='DataRow'))
        for i, rt in enumerate(row_tuples):
            for k in range(self.K):
                U_all[i, k, :], _ = self._build_u_vector(rt, k)

        for k in range(self.K):
            w = resp[:, k]
            p = w / max(w.sum(), 1e-12)
            if np.any(np.isnan(p)) or p.sum() == 0:
                idx = self.random_state.integers(0, n, size=min(self.corr_subsample, n))
            else:
                m = min(self.corr_subsample, n)
                idx = self.random_state.choice(n, size=m, replace=False, p=p)

            Z = norm.ppf(np.clip(U_all[idx, k, :], 1e-10, 1 - 1e-10))
            
            R = self._pairwise_weighted_corr(Z, np.ones(len(idx)))
            R = _shrink_corr(R, self.shrink_lambda) if self.shrink_lambda > 0 else _nearest_pd(R)
            
            diag = np.sqrt(np.diag(R))
            if not np.all(diag > 1e-9): continue
            R /= np.outer(diag, diag)
            np.fill_diagonal(R, 1.0)
            self.R_[k] = R

            Rabs = np.abs(R); np.fill_diagonal(Rabs, 0.0)
            if self.speedy_graph == 'mst':
                self.speedy_edges_[k] = _max_spanning_tree_from_corr(Rabs)
            else: # knn
                self.speedy_edges_[k] = _knn_graph_edges(Rabs, k_per_node=self.speedy_k_per_node)
    
    def _M_step(self, df, resp):
        """Combined M-step for Speedy mode."""
        self._M_step_marginals(df, resp)
        self._M_step_copulas(df, resp)

    def _log_pk_row_speedy(self, row_tuple):
        log_pk = np.zeros(self.K)
        all_cols_map = {c: i for i, c in enumerate(self.cont_cols_ + self.cat_cols_ + self.ord_cols_)}
        obs_idx = {all_cols_map[c] for c in row_tuple._fields if not pd.isna(getattr(row_tuple, c))}
        
        for k in range(self.K):
            u, log_marg = self._build_u_vector(row_tuple, k)
            if not obs_idx: continue

            edges = self.speedy_edges_[k]
            if not edges: log_c = 0.0
            else:
                msum, wsum = 0.0, 0.0
                Rk = self.R_[k]
                for (i, j) in edges:
                    if i in obs_idx and j in obs_idx:
                        ui, uj = u[i], u[j]
                        if np.isnan(ui) or np.isnan(uj): continue
                        rho = Rk[i, j]
                        w = abs(rho) if self.pairwise_weight == 'abs_rho' else 1.0
                        msum += w * _log_bivariate_gaussian_copula(ui, uj, rho)
                        wsum += w
                log_c = msum / max(wsum, 1e-9)
            log_pk[k] = log_marg + log_c
        return log_pk

    def _E_step(self, df):
        n = len(df)
        B = max(1, self.e_step_batch)
        log_resp = np.zeros((n, self.K))
        row_tuples = list(df.itertuples(index=False, name='DataRow'))
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_batch)(row_tuples[s:min(n, s+B)]) for s in range(0, n, B)
        )
        
        for s, blk in zip(range(0, n, B), results):
            log_resp[s:min(n, s+B)] = blk
            
        log_resp += _safe_log(self.pi_)
        ll_per_sample = logsumexp(log_resp, axis=1)
        loglik = np.sum(ll_per_sample)
        log_resp -= ll_per_sample[:, None]
        return np.exp(log_resp), loglik

    def _process_batch(self, batch_tuples):
        return np.array([self._log_pk_row_speedy(rt) for rt in batch_tuples])

