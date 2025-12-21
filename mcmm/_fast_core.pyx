# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False

"""
pymcmm Cython高速化コア実装 v0.3.0

E-step, M-step全体をバッチ処理で高速化。
scipy.statsを使わず、数学関数を直接実装することで大幅な高速化を実現。

コンパイル:
    python setup.py build_ext --inplace
"""

import numpy as np
cimport numpy as np
from libc.math cimport log, sqrt, exp, fabs, M_PI, lgamma, pow, erf, isnan, INFINITY
from libc.stdlib cimport malloc, free
from cython.parallel import prange
cimport cython

# 型定義
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t INT_t
ctypedef np.int32_t INT32_t

# 定数
cdef double LOG_2PI = 1.8378770664093453  # log(2*pi)
cdef double SQRT_2 = 1.4142135623730951
cdef double INV_SQRT_2 = 0.7071067811865476
cdef double EPS = 1e-12
cdef double CLIP_LO = 1e-10
cdef double CLIP_HI = 0.9999999999

# ============================================================
# 基本ユーティリティ関数
# ============================================================

cdef inline double safe_log(double x) noexcept nogil:
    """安全な対数（log(0)を回避）"""
    if x < EPS:
        return log(EPS)
    return log(x)


cdef inline double clip_double(double x, double lo, double hi) noexcept nogil:
    """値のクリッピング"""
    if x < lo:
        return lo
    elif x > hi:
        return hi
    return x


cdef inline double clip_prob(double x) noexcept nogil:
    """確率値のクリッピング [1e-10, 1-1e-10]"""
    if x < CLIP_LO:
        return CLIP_LO
    elif x > CLIP_HI:
        return CLIP_HI
    return x


# ============================================================
# 正規分布関数（scipy不要の実装）
# ============================================================

cdef inline double norm_cdf(double x) noexcept nogil:
    """
    標準正規分布のCDF
    erfc(誤差補関数)を使用した高精度実装
    Φ(x) = 0.5 * erfc(-x / sqrt(2))
    """
    # 定数
    cdef double INV_SQRT2 = 0.7071067811865476  # 1/sqrt(2)
    
    # A&S 7.1.26 による erfc の近似
    # erfc(x) ≈ t * exp(-x^2 + P(t)) for x >= 0
    # where t = 1/(1 + 0.5*x)
    
    cdef double a1 =  0.254829592
    cdef double a2 = -0.284496736
    cdef double a3 =  1.421413741
    cdef double a4 = -1.453152027
    cdef double a5 =  1.061405429
    cdef double p  =  0.3275911
    
    cdef double ax = x * INV_SQRT2  # x / sqrt(2) for erf transformation
    cdef int sign = 1
    
    if ax < 0:
        sign = -1
        ax = -ax
    
    # erf(|ax|) using A&S 7.1.26
    cdef double t = 1.0 / (1.0 + p * ax)
    cdef double erf_val = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-ax * ax)
    
    # erf(-x) = -erf(x)
    if sign < 0:
        erf_val = -erf_val
    
    # Φ(x) = 0.5 * (1 + erf(x/sqrt(2)))
    return 0.5 * (1.0 + erf_val)


cdef inline double norm_ppf(double p) noexcept nogil:
    """
    標準正規分布の逆CDF（分位点関数）
    Rational approximation by Peter J. Acklam
    最大相対誤差 < 1.15e-9
    """
    if p <= 0.0:
        return -37.0
    if p >= 1.0:
        return 37.0
    if p == 0.5:
        return 0.0
    
    # 係数
    cdef double a1 = -3.969683028665376e+01
    cdef double a2 =  2.209460984245205e+02
    cdef double a3 = -2.759285104469687e+02
    cdef double a4 =  1.383577518672690e+02
    cdef double a5 = -3.066479806614716e+01
    cdef double a6 =  2.506628277459239e+00
    
    cdef double b1 = -5.447609879822406e+01
    cdef double b2 =  1.615858368580409e+02
    cdef double b3 = -1.556989798598866e+02
    cdef double b4 =  6.680131188771972e+01
    cdef double b5 = -1.328068155288572e+01
    
    cdef double c1 = -7.784894002430293e-03
    cdef double c2 = -3.223964580411365e-01
    cdef double c3 = -2.400758277161838e+00
    cdef double c4 = -2.549732539343734e+00
    cdef double c5 =  4.374664141464968e+00
    cdef double c6 =  2.938163982698783e+00
    
    cdef double d1 =  7.784695709041462e-03
    cdef double d2 =  3.224671290700398e-01
    cdef double d3 =  2.445134137142996e+00
    cdef double d4 =  3.754408661907416e+00
    
    cdef double p_low = 0.02425
    cdef double p_high = 1.0 - p_low
    cdef double q, r, result
    
    if p < p_low:
        # 左側の尾
        q = sqrt(-2.0 * log(p))
        result = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
                 ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    elif p <= p_high:
        # 中央領域
        q = p - 0.5
        r = q * q
        result = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q / \
                 (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
    else:
        # 右側の尾
        q = sqrt(-2.0 * log(1.0 - p))
        result = -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / \
                  ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
    
    return result


cdef inline double norm_logpdf(double x) noexcept nogil:
    """標準正規分布の対数密度"""
    return -0.5 * x * x - 0.5 * LOG_2PI


cdef inline double gaussian_logpdf(double x, double mu, double sig) noexcept nogil:
    """正規分布の対数密度"""
    cdef double z
    if sig < 1e-9:
        sig = 1e-9
    z = (x - mu) / sig
    return -0.5 * z * z - safe_log(sig) - 0.5 * LOG_2PI


cdef inline double gaussian_cdf(double x, double mu, double sig) noexcept nogil:
    """正規分布のCDF"""
    if sig < 1e-9:
        sig = 1e-9
    return norm_cdf((x - mu) / sig)


# ============================================================
# Student-t分布関数
# ============================================================

cdef inline double _betacf(double a, double b, double x) noexcept nogil:
    """
    連分数展開によるベータ関数の計算
    Numerical Recipes準拠
    """
    cdef int MAXIT = 200
    cdef double EPS_CF = 3.0e-12
    cdef double FPMIN = 1.0e-30
    
    cdef double qab = a + b
    cdef double qap = a + 1.0
    cdef double qam = a - 1.0
    cdef double c = 1.0
    cdef double d = 1.0 - qab * x / qap
    
    if fabs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    cdef double h = d
    
    cdef int m
    cdef double m2, aa, del_val
    
    for m in range(1, MAXIT + 1):
        m2 = 2.0 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if fabs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if fabs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h = h * d * c
        
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if fabs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if fabs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        del_val = d * c
        h = h * del_val
        
        if fabs(del_val - 1.0) < EPS_CF:
            break
    
    return h


cdef inline double _betai(double a, double b, double x) noexcept nogil:
    """
    不完全ベータ関数 I_x(a, b)
    """
    if x < 0.0 or x > 1.0:
        return 0.0
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    
    cdef double bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + 
                        a * log(x) + b * log(1.0 - x))
    
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    else:
        return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


cdef inline double studentt_cdf(double t, double nu) noexcept nogil:
    """
    Student-t分布のCDF
    不完全ベータ関数を使用
    """
    # 自由度が大きい場合は正規近似
    if nu > 100.0:
        return norm_cdf(t)
    
    cdef double x = nu / (nu + t * t)
    cdef double p = 0.5 * _betai(nu / 2.0, 0.5, x)
    
    if t >= 0:
        return 1.0 - p
    else:
        return p


cdef inline double studentt_logpdf(double x, double mu, double sig, double nu) noexcept nogil:
    """
    Student-t分布の対数密度
    """
    if sig < 1e-9:
        sig = 1e-9
    
    cdef double z = (x - mu) / sig
    cdef double z2 = z * z
    
    # log-gamma関数を使用
    cdef double log_norm = lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0) - 0.5 * log(nu * M_PI)
    cdef double log_kernel = -((nu + 1.0) / 2.0) * log(1.0 + z2 / nu)
    
    return log_norm + log_kernel - safe_log(sig)


cdef inline double studentt_cdf_scaled(double x, double mu, double sig, double nu) noexcept nogil:
    """
    スケール付きStudent-t分布のCDF
    """
    if sig < 1e-9:
        sig = 1e-9
    cdef double z = (x - mu) / sig
    return studentt_cdf(z, nu)


# ============================================================
# 二変量ガウスコピュラ
# ============================================================

cdef inline double log_bivariate_gaussian_copula(double u1, double u2, double rho) noexcept nogil:
    """
    二変量ガウスコピュラの対数密度
    
    c(u1, u2; rho) = exp(-0.5 * log(1-rho^2) - (z1^2 + z2^2 - 2*rho*z1*z2)/(2*(1-rho^2)) + 0.5*(z1^2+z2^2))
    
    where z1 = Phi^{-1}(u1), z2 = Phi^{-1}(u2)
    """
    # クリッピング
    u1 = clip_prob(u1)
    u2 = clip_prob(u2)
    rho = clip_double(rho, -0.999999, 0.999999)
    
    cdef double z1 = norm_ppf(u1)
    cdef double z2 = norm_ppf(u2)
    cdef double r2 = rho * rho
    cdef double one_minus_r2 = 1.0 - r2
    
    cdef double log_det_term = -0.5 * log(one_minus_r2)
    cdef double quad_term = (z1*z1 + z2*z2 - 2.0*rho*z1*z2) / (2.0 * one_minus_r2)
    cdef double marginal_term = 0.5 * (z1*z1 + z2*z2)
    
    return log_det_term - quad_term + marginal_term


# ============================================================
# Python呼び出し用ラッパー関数
# ============================================================

def py_norm_cdf(double x):
    """Pythonから呼び出し可能な標準正規CDF"""
    return norm_cdf(x)

def py_norm_ppf(double p):
    """Pythonから呼び出し可能な標準正規逆CDF"""
    return norm_ppf(p)

def py_studentt_cdf(double x, double nu):
    """Pythonから呼び出し可能なt分布CDF"""
    return studentt_cdf(x, nu)

def py_studentt_logpdf(double x, double mu, double sig, double nu):
    """Pythonから呼び出し可能なt分布対数密度"""
    return studentt_logpdf(x, mu, sig, nu)

def py_log_bivariate_gaussian_copula(double u1, double u2, double rho):
    """Pythonから呼び出し可能な二変量ガウスコピュラ"""
    return log_bivariate_gaussian_copula(u1, u2, rho)


# ============================================================
# ベクトル化された計算関数
# ============================================================

def studentt_cdf_array(np.ndarray[DTYPE_t, ndim=1] x not None,
                       double mu, double sig, double nu):
    """
    Student-t CDF のベクトル化版
    
    Parameters
    ----------
    x : ndarray (n,)
        入力値
    mu : float
        位置パラメータ
    sig : float
        スケールパラメータ
    nu : float
        自由度
    
    Returns
    -------
    ndarray (n,)
        CDF値
    """
    cdef int n = x.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.empty(n, dtype=DTYPE)
    cdef int i
    cdef double z, s = sig
    
    if s < 1e-9:
        s = 1e-9
    
    for i in prange(n, nogil=True):
        z = (x[i] - mu) / s
        result[i] = studentt_cdf(z, nu)
    
    return result


def studentt_logpdf_array(np.ndarray[DTYPE_t, ndim=1] x not None,
                          double mu, double sig, double nu):
    """
    Student-t 対数密度のベクトル化版
    """
    cdef int n = x.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.empty(n, dtype=DTYPE)
    cdef int i
    
    for i in prange(n, nogil=True):
        result[i] = studentt_logpdf(x[i], mu, sig, nu)
    
    return result


def gaussian_cdf_array(np.ndarray[DTYPE_t, ndim=1] x not None,
                       double mu, double sig):
    """
    正規分布CDFのベクトル化版
    """
    cdef int n = x.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.empty(n, dtype=DTYPE)
    cdef int i
    cdef double z, s = sig
    
    if s < 1e-9:
        s = 1e-9
    
    for i in prange(n, nogil=True):
        z = (x[i] - mu) / s
        result[i] = norm_cdf(z)
    
    return result


def norm_ppf_array(np.ndarray[DTYPE_t, ndim=1] p not None):
    """
    標準正規逆CDFのベクトル化版
    """
    cdef int n = p.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.empty(n, dtype=DTYPE)
    cdef int i
    
    for i in prange(n, nogil=True):
        result[i] = norm_ppf(clip_prob(p[i]))
    
    return result


# ============================================================
# 連続変数のU値・周辺密度計算（バッチ処理）
# ============================================================

def compute_cont_u_and_logmarg(np.ndarray[DTYPE_t, ndim=2] X_cont not None,
                                np.ndarray[DTYPE_t, ndim=2] mu not None,
                                np.ndarray[DTYPE_t, ndim=2] sig not None,
                                double nu,
                                int K,
                                str marginal_type='student_t'):
    """
    連続変数のU値と周辺対数密度を一括計算
    
    Parameters
    ----------
    X_cont : ndarray (n, p_cont)
        連続変数データ
    mu : ndarray (K, p_cont)
        各クラスタの平均
    sig : ndarray (K, p_cont)
        各クラスタの標準偏差
    nu : float
        t分布の自由度
    K : int
        クラスタ数
    marginal_type : str
        'gaussian' or 'student_t'
    
    Returns
    -------
    U : ndarray (n, K, p_cont)
        CDF値（NaNは-1.0でマーク）
    log_marg : ndarray (n, K)
        周辺対数密度の合計
    """
    cdef int n = X_cont.shape[0]
    cdef int p = X_cont.shape[1]
    cdef int i, j, k
    cdef double x_val, m, s, z
    cdef bint use_t = (marginal_type == 'student_t')
    
    cdef np.ndarray[DTYPE_t, ndim=3] U = np.empty((n, K, p), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] log_marg = np.zeros((n, K), dtype=DTYPE)
    
    for i in prange(n, nogil=True):
        for k in range(K):
            for j in range(p):
                x_val = X_cont[i, j]
                
                # NaNチェック
                if isnan(x_val):
                    U[i, k, j] = -1.0  # NaNマーカー
                    continue
                
                m = mu[k, j]
                s = sig[k, j]
                if s < 1e-9:
                    s = 1e-9
                
                if use_t:
                    U[i, k, j] = studentt_cdf_scaled(x_val, m, s, nu)
                    log_marg[i, k] = log_marg[i, k] + studentt_logpdf(x_val, m, s, nu)
                else:
                    U[i, k, j] = gaussian_cdf(x_val, m, s)
                    log_marg[i, k] = log_marg[i, k] + gaussian_logpdf(x_val, m, s)
    
    return U, log_marg


# ============================================================
# カテゴリ変数のU値計算（バッチ処理）- 新規追加
# ============================================================

def compute_cat_u_and_logmarg(np.ndarray[INT32_t, ndim=2] X_cat_idx not None,
                               list cat_probs_list,
                               int K,
                               str dt_mode='mid'):
    """
    カテゴリ変数のU値と周辺対数密度をバッチ計算
    
    Parameters
    ----------
    X_cat_idx : ndarray (n, n_cat), int32
        カテゴリ変数のインデックス（-1 = 欠損）
    cat_probs_list : list of ndarray
        各カテゴリ変数の確率 [(K, L_c), ...]
    K : int
        クラスタ数
    dt_mode : str
        'mid' のみ対応
    
    Returns
    -------
    U : ndarray (n, K, n_cat)
        U値（NaNは-1.0でマーク）
    log_marg : ndarray (n, K)
        周辺対数密度の合計
    """
    cdef int n = X_cat_idx.shape[0]
    cdef int n_cat = X_cat_idx.shape[1]
    cdef int i, c, k, idx, L
    cdef double prob_val, Fm, F
    
    cdef np.ndarray[DTYPE_t, ndim=3] U = np.empty((n, K, n_cat), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] log_marg = np.zeros((n, K), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] probs
    
    for c in range(n_cat):
        probs = np.asarray(cat_probs_list[c], dtype=DTYPE)
        L = probs.shape[1]
        
        for i in range(n):
            idx = X_cat_idx[i, c]
            
            if idx < 0:  # 欠損値
                for k in range(K):
                    U[i, k, c] = -1.0
                continue
            
            for k in range(K):
                if idx >= L:
                    prob_val = 1e-12
                    Fm = 0.0
                    F = 1.0
                else:
                    prob_val = probs[k, idx]
                    if prob_val < 1e-12:
                        prob_val = 1e-12
                    
                    # Fm = sum(probs[k, :idx])
                    Fm = 0.0
                    for j in range(idx):
                        Fm = Fm + probs[k, j]
                    F = Fm + prob_val
                
                log_marg[i, k] = log_marg[i, k] + safe_log(prob_val)
                U[i, k, c] = 0.5 * (Fm + F)
    
    return U, log_marg


# ============================================================
# E-step完全バッチ処理 - 新規追加
# ============================================================

def compute_e_step_full(np.ndarray[DTYPE_t, ndim=2] X_cont not None,
                         np.ndarray[INT32_t, ndim=2] X_cat_idx not None,
                         np.ndarray[DTYPE_t, ndim=2] mu not None,
                         np.ndarray[DTYPE_t, ndim=2] sig not None,
                         list cat_probs_list,
                         list R_list,
                         np.ndarray[DTYPE_t, ndim=1] pi not None,
                         double nu,
                         str marginal_type='student_t',
                         str weight_type='abs_rho',
                         str copula_type='pairwise'):
    """
    E-step全体をバッチ処理で実行
    
    Parameters
    ----------
    X_cont : ndarray (n, p_cont)
        連続変数データ
    X_cat_idx : ndarray (n, n_cat)
        カテゴリ変数インデックス（-1=欠損）
    mu, sig : ndarray (K, p_cont)
        周辺分布パラメータ
    cat_probs_list : list
        各カテゴリ変数の確率
    R_list : list of ndarray
        各クラスタの相関行列
    pi : ndarray (K,)
        混合比
    nu : float
        t分布の自由度
    marginal_type : str
        'gaussian' or 'student_t'
    weight_type : str
        'uniform' or 'abs_rho'
    copula_type : str
        'full' or 'pairwise'
    
    Returns
    -------
    log_resp : ndarray (n, K)
        対数責任度
    total_loglik : float
        対数尤度合計
    ll_per_sample : ndarray (n,)
        サンプルごとの対数尤度
    """
    cdef int n = X_cont.shape[0]
    cdef int p_cont = X_cont.shape[1]
    cdef int n_cat = X_cat_idx.shape[1]
    cdef int K = mu.shape[0]
    cdef int d_all = p_cont + n_cat
    cdef int i, j, k, c, idx
    cdef double x_val, m, s, prob_val, Fm, F
    cdef double log_c, msum, wsum, ui, uj, rho, w
    cdef bint use_t = (marginal_type == 'student_t')
    cdef bint use_abs_rho = (weight_type == 'abs_rho')
    cdef bint use_full = (copula_type == 'full')
    
    # U値と周辺密度の配列
    cdef np.ndarray[DTYPE_t, ndim=3] U = np.empty((n, K, d_all), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] log_pk = np.zeros((n, K), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] R
    cdef np.ndarray[DTYPE_t, ndim=2] probs
    
    # 1. 連続変数のU値と周辺密度
    for i in prange(n, nogil=True):
        for k in range(K):
            for j in range(p_cont):
                x_val = X_cont[i, j]
                
                if isnan(x_val):
                    U[i, k, j] = -1.0
                    continue
                
                m = mu[k, j]
                s = sig[k, j]
                if s < 1e-9:
                    s = 1e-9
                
                if use_t:
                    U[i, k, j] = studentt_cdf_scaled(x_val, m, s, nu)
                    log_pk[i, k] = log_pk[i, k] + studentt_logpdf(x_val, m, s, nu)
                else:
                    U[i, k, j] = gaussian_cdf(x_val, m, s)
                    log_pk[i, k] = log_pk[i, k] + gaussian_logpdf(x_val, m, s)
    
    # 2. カテゴリ変数のU値と周辺密度（GIL必要）
    for c in range(n_cat):
        probs = np.asarray(cat_probs_list[c], dtype=DTYPE)
        L = probs.shape[1]
        
        for i in range(n):
            idx = X_cat_idx[i, c]
            
            if idx < 0:
                for k in range(K):
                    U[i, k, p_cont + c] = -1.0
                continue
            
            for k in range(K):
                if idx >= L:
                    prob_val = 1e-12
                    Fm = 0.0
                    F = 1.0
                else:
                    prob_val = probs[k, idx]
                    if prob_val < 1e-12:
                        prob_val = 1e-12
                    
                    Fm = 0.0
                    for jj in range(idx):
                        Fm = Fm + probs[k, jj]
                    F = Fm + prob_val
                
                log_pk[i, k] = log_pk[i, k] + safe_log(prob_val)
                U[i, k, p_cont + c] = 0.5 * (Fm + F)
    
    # 3. コピュラ対数尤度
    for k in range(K):
        R = np.asarray(R_list[k], dtype=DTYPE)
        
        for i in range(n):
            msum = 0.0
            wsum = 0.0
            
            for ii in range(d_all):
                ui = U[i, k, ii]
                if ui < 0:
                    continue
                
                for jj in range(ii + 1, d_all):
                    uj = U[i, k, jj]
                    if uj < 0:
                        continue
                    
                    rho = R[ii, jj]
                    
                    if use_abs_rho:
                        w = fabs(rho)
                    else:
                        w = 1.0
                    
                    log_c = log_bivariate_gaussian_copula(ui, uj, rho)
                    msum = msum + w * log_c
                    wsum = wsum + w
            
            if wsum > 1e-9:
                log_pk[i, k] = log_pk[i, k] + msum / wsum
    
    # 4. 混合比を追加
    cdef np.ndarray[DTYPE_t, ndim=2] log_weighted_pk = np.empty((n, K), dtype=DTYPE)
    for i in range(n):
        for k in range(K):
            log_weighted_pk[i, k] = log_pk[i, k] + safe_log(pi[k])
    
    # 5. logsumexp
    cdef np.ndarray[DTYPE_t, ndim=1] ll_per_sample = np.empty(n, dtype=DTYPE)
    cdef double max_val, sum_exp
    
    for i in prange(n, nogil=True):
        max_val = log_weighted_pk[i, 0]
        for k in range(1, K):
            if log_weighted_pk[i, k] > max_val:
                max_val = log_weighted_pk[i, k]
        
        sum_exp = 0.0
        for k in range(K):
            sum_exp = sum_exp + exp(log_weighted_pk[i, k] - max_val)
        
        ll_per_sample[i] = max_val + log(sum_exp)
    
    # 6. 正規化（対数責任度）
    cdef np.ndarray[DTYPE_t, ndim=2] log_resp = np.empty((n, K), dtype=DTYPE)
    for i in range(n):
        for k in range(K):
            log_resp[i, k] = log_weighted_pk[i, k] - ll_per_sample[i]
    
    cdef double total_loglik = 0.0
    for i in range(n):
        total_loglik = total_loglik + ll_per_sample[i]
    
    return log_resp, total_loglik, ll_per_sample


# ============================================================
# E-step (Speedy mode) - 新規追加
# ============================================================

def compute_e_step_speedy(np.ndarray[DTYPE_t, ndim=2] X_cont not None,
                           np.ndarray[INT32_t, ndim=2] X_cat_idx not None,
                           np.ndarray[DTYPE_t, ndim=2] mu not None,
                           np.ndarray[DTYPE_t, ndim=2] sig not None,
                           list cat_probs_list,
                           list R_list,
                           list edges_list,
                           np.ndarray[DTYPE_t, ndim=1] pi not None,
                           double nu,
                           str marginal_type='student_t',
                           str weight_type='abs_rho'):
    """
    Speedy mode用E-step（MST/KNNエッジのみ使用）
    """
    cdef int n = X_cont.shape[0]
    cdef int p_cont = X_cont.shape[1]
    cdef int n_cat = X_cat_idx.shape[1]
    cdef int K = mu.shape[0]
    cdef int d_all = p_cont + n_cat
    cdef int i, j, k, c, idx, e, ei, ej, jj, L
    cdef double x_val, m, s, prob_val, Fm, F
    cdef double log_c, msum, wsum, ui, uj, rho, w
    cdef bint use_t = (marginal_type == 'student_t')
    cdef bint use_abs_rho = (weight_type == 'abs_rho')
    
    cdef np.ndarray[DTYPE_t, ndim=3] U = np.empty((n, K, d_all), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] log_pk = np.zeros((n, K), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] R
    cdef np.ndarray[DTYPE_t, ndim=2] probs
    cdef list edges
    cdef int n_edges
    
    # 1. 連続変数
    for i in prange(n, nogil=True):
        for k in range(K):
            for j in range(p_cont):
                x_val = X_cont[i, j]
                
                if isnan(x_val):
                    U[i, k, j] = -1.0
                    continue
                
                m = mu[k, j]
                s = sig[k, j]
                if s < 1e-9:
                    s = 1e-9
                
                if use_t:
                    U[i, k, j] = studentt_cdf_scaled(x_val, m, s, nu)
                    log_pk[i, k] = log_pk[i, k] + studentt_logpdf(x_val, m, s, nu)
                else:
                    U[i, k, j] = gaussian_cdf(x_val, m, s)
                    log_pk[i, k] = log_pk[i, k] + gaussian_logpdf(x_val, m, s)
    
    # 2. カテゴリ変数
    for c in range(n_cat):
        probs = np.asarray(cat_probs_list[c], dtype=DTYPE)
        L = probs.shape[1]
        
        for i in range(n):
            idx = X_cat_idx[i, c]
            
            if idx < 0:
                for k in range(K):
                    U[i, k, p_cont + c] = -1.0
                continue
            
            for k in range(K):
                if idx >= L:
                    prob_val = 1e-12
                    Fm = 0.0
                    F = 1.0
                else:
                    prob_val = probs[k, idx]
                    if prob_val < 1e-12:
                        prob_val = 1e-12
                    
                    Fm = 0.0
                    for jj in range(idx):
                        Fm = Fm + probs[k, jj]
                    F = Fm + prob_val
                
                log_pk[i, k] = log_pk[i, k] + safe_log(prob_val)
                U[i, k, p_cont + c] = 0.5 * (Fm + F)
    
    # 3. コピュラ対数尤度（エッジリストのみ）
    for k in range(K):
        R = np.asarray(R_list[k], dtype=DTYPE)
        edges = edges_list[k]
        n_edges = len(edges)
        
        if n_edges == 0:
            continue
        
        for i in range(n):
            msum = 0.0
            wsum = 0.0
            
            for e in range(n_edges):
                ei, ej = edges[e]
                ui = U[i, k, ei]
                uj = U[i, k, ej]
                
                if ui < 0 or uj < 0:
                    continue
                
                rho = R[ei, ej]
                
                if use_abs_rho:
                    w = fabs(rho)
                else:
                    w = 1.0
                
                log_c = log_bivariate_gaussian_copula(ui, uj, rho)
                msum = msum + w * log_c
                wsum = wsum + w
            
            if wsum > 1e-9:
                log_pk[i, k] = log_pk[i, k] + msum / wsum
    
    # 4. 混合比を追加
    cdef np.ndarray[DTYPE_t, ndim=2] log_weighted_pk = np.empty((n, K), dtype=DTYPE)
    for i in range(n):
        for k in range(K):
            log_weighted_pk[i, k] = log_pk[i, k] + safe_log(pi[k])
    
    # 5. logsumexp
    cdef np.ndarray[DTYPE_t, ndim=1] ll_per_sample = np.empty(n, dtype=DTYPE)
    cdef double max_val, sum_exp
    
    for i in prange(n, nogil=True):
        max_val = log_weighted_pk[i, 0]
        for k in range(1, K):
            if log_weighted_pk[i, k] > max_val:
                max_val = log_weighted_pk[i, k]
        
        sum_exp = 0.0
        for k in range(K):
            sum_exp = sum_exp + exp(log_weighted_pk[i, k] - max_val)
        
        ll_per_sample[i] = max_val + log(sum_exp)
    
    # 6. 正規化
    cdef np.ndarray[DTYPE_t, ndim=2] log_resp = np.empty((n, K), dtype=DTYPE)
    for i in range(n):
        for k in range(K):
            log_resp[i, k] = log_weighted_pk[i, k] - ll_per_sample[i]
    
    cdef double total_loglik = 0.0
    for i in range(n):
        total_loglik = total_loglik + ll_per_sample[i]
    
    return log_resp, total_loglik, ll_per_sample


# ============================================================
# M-step: 周辺分布パラメータ更新 - 新規追加
# ============================================================

def compute_m_step_marginals_cont(np.ndarray[DTYPE_t, ndim=2] X_cont not None,
                                   np.ndarray[DTYPE_t, ndim=2] resp not None,
                                   np.ndarray[DTYPE_t, ndim=2] mu_old not None,
                                   np.ndarray[DTYPE_t, ndim=2] sig_old not None,
                                   double nu,
                                   str marginal_type='student_t'):
    """
    M-step: 連続変数の周辺分布パラメータ更新
    
    Parameters
    ----------
    X_cont : ndarray (n, p_cont)
    resp : ndarray (n, K)
    mu_old, sig_old : ndarray (K, p_cont)
    nu : float
    marginal_type : str
    
    Returns
    -------
    mu_new, sig_new : ndarray (K, p_cont)
    all_z, all_w : list
        t分布のnu推定用（estimate_nu=Trueの場合）
    """
    cdef int n = X_cont.shape[0]
    cdef int p = X_cont.shape[1]
    cdef int K = resp.shape[1]
    cdef int i, j, k
    cdef double x_val, w_em, z, w_t, w_sum, mu, var
    cdef bint use_t = (marginal_type == 'student_t')
    
    cdef np.ndarray[DTYPE_t, ndim=2] mu_new = np.zeros((K, p), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] sig_new = np.ones((K, p), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] w_arr = np.empty(n, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] x_arr = np.empty(n, dtype=DTYPE)
    
    all_z = []
    all_w = []
    
    for k in range(K):
        for j in range(p):
            # 有効なサンプルを収集
            w_sum = 0.0
            mu = 0.0
            cnt = 0
            
            for i in range(n):
                x_val = X_cont[i, j]
                if isnan(x_val):
                    continue
                
                w_em = resp[i, k]
                
                if use_t:
                    z = (x_val - mu_old[k, j]) / max(sig_old[k, j], 1e-9)
                    w_t = (nu + 1.0) / (nu + z * z)
                    w_em = w_em * w_t
                
                w_arr[cnt] = w_em
                x_arr[cnt] = x_val
                w_sum = w_sum + w_em
                mu = mu + w_em * x_val
                cnt += 1
            
            if w_sum < 1e-9 or cnt == 0:
                mu_new[k, j] = mu_old[k, j]
                sig_new[k, j] = sig_old[k, j]
                continue
            
            mu = mu / w_sum
            
            # 分散計算
            var = 0.0
            for i in range(cnt):
                var = var + w_arr[i] * (x_arr[i] - mu) * (x_arr[i] - mu)
            var = var / w_sum
            
            mu_new[k, j] = mu
            sig_new[k, j] = sqrt(max(var, 1e-10))
            
            # nu推定用のzとwを保存
            if use_t:
                z_list = []
                w_list = []
                for i in range(cnt):
                    z = (x_arr[i] - mu) / max(sig_new[k, j], 1e-9)
                    z_list.append(z)
                    w_list.append(w_arr[i])
                if z_list:
                    all_z.append(np.array(z_list, dtype=DTYPE))
                    all_w.append(np.array(w_list, dtype=DTYPE))
    
    return mu_new, sig_new, all_z, all_w


def compute_m_step_marginals_cat(np.ndarray[INT32_t, ndim=2] X_cat_idx not None,
                                  np.ndarray[DTYPE_t, ndim=2] resp not None,
                                  list n_levels_list):
    """
    M-step: カテゴリ変数の確率更新
    
    Parameters
    ----------
    X_cat_idx : ndarray (n, n_cat)
    resp : ndarray (n, K)
    n_levels_list : list of int
        各カテゴリ変数のレベル数
    
    Returns
    -------
    probs_list : list of ndarray (K, L_c)
    """
    cdef int n = X_cat_idx.shape[0]
    cdef int n_cat = X_cat_idx.shape[1]
    cdef int K = resp.shape[1]
    cdef int i, c, k, idx, L
    cdef double w
    
    probs_list = []
    
    for c in range(n_cat):
        L = n_levels_list[c]
        probs = np.zeros((K, L), dtype=DTYPE)
        
        for i in range(n):
            idx = X_cat_idx[i, c]
            if idx < 0 or idx >= L:
                continue
            
            for k in range(K):
                w = resp[i, k]
                probs[k, idx] = probs[k, idx] + w
        
        # 正規化
        for k in range(K):
            row_sum = 0.0
            for l in range(L):
                probs[k, l] = probs[k, l] + 1e-9
                row_sum = row_sum + probs[k, l]
            for l in range(L):
                probs[k, l] = probs[k, l] / row_sum
        
        probs_list.append(probs)
    
    return probs_list


# ============================================================
# M-step: コピュラパラメータ（相関行列）更新 - 新規追加
# ============================================================

def compute_u_matrix(np.ndarray[DTYPE_t, ndim=2] X_cont not None,
                      np.ndarray[INT32_t, ndim=2] X_cat_idx not None,
                      np.ndarray[DTYPE_t, ndim=2] mu not None,
                      np.ndarray[DTYPE_t, ndim=2] sig not None,
                      list cat_probs_list,
                      int k,
                      double nu,
                      str marginal_type='student_t'):
    """
    単一クラスタkのU行列を計算
    
    Parameters
    ----------
    X_cont : ndarray (n, p_cont)
    X_cat_idx : ndarray (n, n_cat)
    mu, sig : ndarray (K, p_cont)
    cat_probs_list : list
    k : int
        クラスタインデックス
    nu : float
    marginal_type : str
    
    Returns
    -------
    U : ndarray (n, d_all)
        U値（NaNは np.nan）
    """
    cdef int n = X_cont.shape[0]
    cdef int p_cont = X_cont.shape[1]
    cdef int n_cat = X_cat_idx.shape[1]
    cdef int d_all = p_cont + n_cat
    cdef int i, j, c, idx, jj, L
    cdef double x_val, m, s, prob_val, Fm, F
    cdef bint use_t = (marginal_type == 'student_t')
    
    cdef np.ndarray[DTYPE_t, ndim=2] U = np.empty((n, d_all), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] probs
    
    # 連続変数
    for i in prange(n, nogil=True):
        for j in range(p_cont):
            x_val = X_cont[i, j]
            
            if isnan(x_val):
                U[i, j] = -1.0  # NaN marker
                continue
            
            m = mu[k, j]
            s = sig[k, j]
            if s < 1e-9:
                s = 1e-9
            
            if use_t:
                U[i, j] = studentt_cdf_scaled(x_val, m, s, nu)
            else:
                U[i, j] = gaussian_cdf(x_val, m, s)
    
    # カテゴリ変数
    for c in range(n_cat):
        probs = np.asarray(cat_probs_list[c], dtype=DTYPE)
        L = probs.shape[1]
        
        for i in range(n):
            idx = X_cat_idx[i, c]
            
            if idx < 0:
                U[i, p_cont + c] = np.nan
                continue
            
            if idx >= L:
                Fm = 0.0
                F = 1.0
            else:
                Fm = 0.0
                for jj in range(idx):
                    Fm = Fm + probs[k, jj]
                F = Fm + probs[k, idx]
            
            U[i, p_cont + c] = 0.5 * (Fm + F)
    
    # -1.0 を NaN に変換
    U[U < 0] = np.nan
    
    return U


# ============================================================
# 加重相関行列の計算
# ============================================================

def compute_weighted_corr(np.ndarray[DTYPE_t, ndim=2] Z not None,
                          np.ndarray[DTYPE_t, ndim=1] weights not None):
    """
    加重相関行列の計算
    
    Parameters
    ----------
    Z : ndarray (n, p)
        標準化データ（NaNあり可）
    weights : ndarray (n,)
        重み
    
    Returns
    -------
    R : ndarray (p, p)
        相関行列
    """
    cdef int n = Z.shape[0]
    cdef int p = Z.shape[1]
    cdef int i, j, idx
    cdef double w_sum, mu_i, mu_j, var_i, var_j, cov, rho
    cdef double zi, zj, wi
    cdef bint valid
    
    cdef np.ndarray[DTYPE_t, ndim=2] R = np.eye(p, dtype=DTYPE)
    
    for i in range(p):
        for j in range(i + 1, p):
            # 両方の変数が観測されているサンプルのみ使用
            w_sum = 0.0
            mu_i = 0.0
            mu_j = 0.0
            
            # 第1パス: 平均計算
            for idx in range(n):
                zi = Z[idx, i]
                zj = Z[idx, j]
                
                if isnan(zi) or isnan(zj):
                    continue
                
                wi = weights[idx]
                w_sum = w_sum + wi
                mu_i = mu_i + wi * zi
                mu_j = mu_j + wi * zj
            
            if w_sum < 1e-9:
                R[i, j] = 0.0
                R[j, i] = 0.0
                continue
            
            mu_i = mu_i / w_sum
            mu_j = mu_j / w_sum
            
            # 第2パス: 分散・共分散計算
            var_i = 0.0
            var_j = 0.0
            cov = 0.0
            
            for idx in range(n):
                zi = Z[idx, i]
                zj = Z[idx, j]
                
                if isnan(zi) or isnan(zj):
                    continue
                
                wi = weights[idx]
                var_i = var_i + wi * (zi - mu_i) * (zi - mu_i)
                var_j = var_j + wi * (zj - mu_j) * (zj - mu_j)
                cov = cov + wi * (zi - mu_i) * (zj - mu_j)
            
            var_i = var_i / w_sum
            var_j = var_j / w_sum
            cov = cov / w_sum
            
            if var_i > 1e-9 and var_j > 1e-9:
                rho = cov / sqrt(var_i * var_j)
                rho = clip_double(rho, -0.999, 0.999)
            else:
                rho = 0.0
            
            R[i, j] = rho
            R[j, i] = rho
    
    return R


# ============================================================
# ペアワイズコピュラ対数尤度
# ============================================================

def compute_pairwise_copula_loglik(np.ndarray[DTYPE_t, ndim=1] u not None,
                                   np.ndarray[DTYPE_t, ndim=2] R not None,
                                   str weight_type='abs_rho'):
    """
    ペアワイズガウスコピュラの対数尤度（加重平均）
    
    Parameters
    ----------
    u : ndarray (p,)
        各変数のCDF値（NaNは-1.0でマーク）
    R : ndarray (p, p)
        相関行列
    weight_type : str
        'uniform' or 'abs_rho'
    
    Returns
    -------
    float
        加重平均対数尤度
    """
    cdef int p = u.shape[0]
    cdef int i, j
    cdef double total = 0.0
    cdef double weight_sum = 0.0
    cdef double rho, w, log_c
    cdef double ui, uj
    cdef bint use_abs_rho = (weight_type == 'abs_rho')
    
    if p <= 1:
        return 0.0
    
    for i in range(p):
        ui = u[i]
        if ui < 0:  # NaNマーカー
            continue
        
        for j in range(i + 1, p):
            uj = u[j]
            if uj < 0:  # NaNマーカー
                continue
            
            rho = R[i, j]
            
            if use_abs_rho:
                w = fabs(rho)
            else:
                w = 1.0
            
            log_c = log_bivariate_gaussian_copula(ui, uj, rho)
            total = total + w * log_c
            weight_sum = weight_sum + w
    
    if weight_sum < 1e-9:
        return 0.0
    
    return total / weight_sum


def compute_pairwise_copula_loglik_edges(np.ndarray[DTYPE_t, ndim=1] u not None,
                                         np.ndarray[DTYPE_t, ndim=2] R not None,
                                         list edges,
                                         str weight_type='abs_rho'):
    """
    指定エッジのみを使用したペアワイズコピュラ対数尤度（Speedy mode用）
    
    Parameters
    ----------
    u : ndarray (p,)
        CDF値
    R : ndarray (p, p)
        相関行列
    edges : list of tuple
        (i, j)のエッジリスト
    weight_type : str
        重み付けタイプ
    """
    cdef int n_edges = len(edges)
    cdef int ei, ej
    cdef double total = 0.0
    cdef double weight_sum = 0.0
    cdef double rho, w, log_c
    cdef double ui, uj
    cdef bint use_abs_rho = (weight_type == 'abs_rho')
    
    if n_edges == 0:
        return 0.0
    
    for edge in edges:
        ei, ej = edge
        ui = u[ei]
        uj = u[ej]
        
        # NaNチェック
        if ui < 0 or uj < 0:
            continue
        if isnan(ui) or isnan(uj):
            continue
        
        rho = R[ei, ej]
        
        if use_abs_rho:
            w = fabs(rho)
        else:
            w = 1.0
        
        log_c = log_bivariate_gaussian_copula(ui, uj, rho)
        total = total + w * log_c
        weight_sum = weight_sum + w
    
    if weight_sum < 1e-9:
        return 0.0
    
    return total / weight_sum


# ============================================================
# E-stepバッチ計算（後方互換性）
# ============================================================

def compute_log_pk_batch_full(np.ndarray[DTYPE_t, ndim=3] U not None,
                              np.ndarray[DTYPE_t, ndim=2] log_marg not None,
                              list R_list,
                              str weight_type='abs_rho'):
    """
    全観測の log P(x|k) をバッチ計算（full pairwise）
    
    Parameters
    ----------
    U : ndarray (n, K, p)
        CDF値（-1はNAN）
    log_marg : ndarray (n, K)
        周辺対数密度
    R_list : list of ndarray
        各クラスタの相関行列 (K, p, p)
    weight_type : str
        'uniform' or 'abs_rho'
    
    Returns
    -------
    log_pk : ndarray (n, K)
        各観測・各クラスタの対数確率
    """
    cdef int n = U.shape[0]
    cdef int K = U.shape[1]
    cdef int p = U.shape[2]
    cdef int i, j, k, idx
    cdef double log_c, msum, wsum
    cdef double ui, uj, rho, w
    cdef bint use_abs_rho = (weight_type == 'abs_rho')
    
    cdef np.ndarray[DTYPE_t, ndim=2] log_pk = log_marg.copy()
    cdef np.ndarray[DTYPE_t, ndim=2] R
    
    for k in range(K):
        R = np.asarray(R_list[k], dtype=DTYPE)
        
        for idx in range(n):
            msum = 0.0
            wsum = 0.0
            
            for i in range(p):
                ui = U[idx, k, i]
                if ui < 0:  # NaN
                    continue
                
                for j in range(i + 1, p):
                    uj = U[idx, k, j]
                    if uj < 0:  # NaN
                        continue
                    
                    rho = R[i, j]
                    
                    if use_abs_rho:
                        w = fabs(rho)
                    else:
                        w = 1.0
                    
                    log_c = log_bivariate_gaussian_copula(ui, uj, rho)
                    msum = msum + w * log_c
                    wsum = wsum + w
            
            if wsum > 1e-9:
                log_pk[idx, k] = log_pk[idx, k] + msum / wsum
    
    return log_pk


def compute_log_pk_batch_speedy(np.ndarray[DTYPE_t, ndim=3] U not None,
                                np.ndarray[DTYPE_t, ndim=2] log_marg not None,
                                list R_list,
                                list edges_list,
                                str weight_type='abs_rho'):
    """
    Speedy mode用：エッジリストを使用したlog P(x|k)計算
    
    Parameters
    ----------
    U : ndarray (n, K, p)
    log_marg : ndarray (n, K)
    R_list : list of ndarray (K個)
    edges_list : list of list of tuple (K個)
    weight_type : str
    """
    cdef int n = U.shape[0]
    cdef int K = U.shape[1]
    cdef int p = U.shape[2]
    cdef int idx, k, e
    cdef int ei, ej
    cdef double log_c, msum, wsum
    cdef double ui, uj, rho, w
    cdef bint use_abs_rho = (weight_type == 'abs_rho')
    
    cdef np.ndarray[DTYPE_t, ndim=2] log_pk = log_marg.copy()
    cdef np.ndarray[DTYPE_t, ndim=2] R
    cdef list edges
    cdef int n_edges
    
    for k in range(K):
        R = np.asarray(R_list[k], dtype=DTYPE)
        edges = edges_list[k]
        n_edges = len(edges)
        
        if n_edges == 0:
            continue
        
        for idx in range(n):
            msum = 0.0
            wsum = 0.0
            
            for e in range(n_edges):
                ei, ej = edges[e]
                ui = U[idx, k, ei]
                uj = U[idx, k, ej]
                
                # NaNチェック
                if ui < 0 or uj < 0:
                    continue
                
                rho = R[ei, ej]
                
                if use_abs_rho:
                    w = fabs(rho)
                else:
                    w = 1.0
                
                log_c = log_bivariate_gaussian_copula(ui, uj, rho)
                msum = msum + w * log_c
                wsum = wsum + w
            
            if wsum > 1e-9:
                log_pk[idx, k] = log_pk[idx, k] + msum / wsum
    
    return log_pk


# ============================================================
# logsumexp (scipy不要)
# ============================================================

def fast_logsumexp(np.ndarray[DTYPE_t, ndim=2] x not None, int axis=1):
    """
    高速logsumexp実装
    
    Parameters
    ----------
    x : ndarray (n, K)
    axis : int
        1 (行方向) のみ対応
    
    Returns
    -------
    ndarray (n,)
    """
    cdef int n = x.shape[0]
    cdef int K = x.shape[1]
    cdef int i, k
    cdef double max_val, sum_exp
    
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.empty(n, dtype=DTYPE)
    
    for i in prange(n, nogil=True):
        # 最大値を見つける
        max_val = x[i, 0]
        for k in range(1, K):
            if x[i, k] > max_val:
                max_val = x[i, k]
        
        # exp の和を計算
        sum_exp = 0.0
        for k in range(K):
            sum_exp = sum_exp + exp(x[i, k] - max_val)
        
        result[i] = max_val + log(sum_exp)
    
    return result


# ============================================================
# 最大全域木 (MST) - Prim法
# ============================================================

def max_spanning_tree(np.ndarray[DTYPE_t, ndim=2] Rabs not None):
    """
    相関行列の絶対値から最大全域木を構築（Prim法）
    
    Parameters
    ----------
    Rabs : ndarray (p, p)
        相関行列の絶対値（対角は0）
    
    Returns
    -------
    list of tuple
        エッジリスト [(i, j), ...]
    """
    cdef int p = Rabs.shape[0]
    if p == 0:
        return []
    
    cdef np.ndarray[np.uint8_t, ndim=1] in_tree = np.zeros(p, dtype=np.uint8)
    in_tree[0] = 1
    
    cdef list edges = []
    cdef int iteration, best_u, best_v, u, v
    cdef double best_w, w
    
    for iteration in range(p - 1):
        best_u = -1
        best_v = -1
        best_w = -1.0
        
        # 木に含まれる頂点から、含まれない頂点への最大重みエッジを探索
        for u in range(p):
            if in_tree[u] == 0:
                continue
            for v in range(p):
                if in_tree[v] == 1:
                    continue
                w = Rabs[u, v]
                if w > best_w:
                    best_w = w
                    best_u = u
                    best_v = v
        
        if best_v == -1:
            break
        
        in_tree[best_v] = 1
        edges.append((best_u, best_v))
    
    return edges


# ============================================================
# ベンチマーク関数
# ============================================================

def benchmark():
    """Cython高速化のベンチマーク"""
    import time
    from scipy.stats import norm as scipy_norm, t as scipy_t
    
    print("=" * 60)
    print("pymcmm Cython高速化ベンチマーク v0.3.0")
    print("=" * 60)
    
    np.random.seed(42)
    n = 10000
    
    # 1. 正規CDF
    print("\n1. 標準正規CDF (n=10,000)")
    x = np.random.randn(n).astype(np.float64)
    
    t0 = time.time()
    for _ in range(10):
        result_scipy = scipy_norm.cdf(x)
    time_scipy = (time.time() - t0) / 10
    
    t0 = time.time()
    for _ in range(10):
        result_cython = np.array([norm_cdf(xi) for xi in x])
    time_cython = (time.time() - t0) / 10
    
    print(f"   scipy:  {time_scipy*1000:.2f} ms")
    print(f"   Cython: {time_cython*1000:.2f} ms")
    print(f"   高速化: {time_scipy/time_cython:.1f}x")
    print(f"   最大誤差: {np.max(np.abs(result_scipy - result_cython)):.2e}")
    
    # 2. Student-t CDF
    print("\n2. Student-t CDF (n=10,000, nu=5)")
    nu = 5.0
    
    t0 = time.time()
    for _ in range(10):
        result_scipy = scipy_t.cdf(x, df=nu)
    time_scipy = (time.time() - t0) / 10
    
    t0 = time.time()
    for _ in range(10):
        result_cython = studentt_cdf_array(x, 0.0, 1.0, nu)
    time_cython = (time.time() - t0) / 10
    
    print(f"   scipy:  {time_scipy*1000:.2f} ms")
    print(f"   Cython: {time_cython*1000:.2f} ms")
    print(f"   高速化: {time_scipy/time_cython:.1f}x")
    print(f"   最大誤差: {np.max(np.abs(result_scipy - result_cython)):.2e}")
    
    # 3. 二変量コピュラ
    print("\n3. 二変量ガウスコピュラ (n=10,000)")
    u1 = np.random.uniform(0.01, 0.99, n).astype(np.float64)
    u2 = np.random.uniform(0.01, 0.99, n).astype(np.float64)
    rho = 0.5
    
    t0 = time.time()
    for _ in range(5):
        for i in range(n):
            z1 = scipy_norm.ppf(u1[i])
            z2 = scipy_norm.ppf(u2[i])
            r2 = rho * rho
            _ = -0.5 * np.log(1-r2) - (z1**2+z2**2-2*rho*z1*z2)/(2*(1-r2)) + 0.5*(z1**2+z2**2)
    time_python = (time.time() - t0) / 5
    
    t0 = time.time()
    for _ in range(5):
        for i in range(n):
            _ = log_bivariate_gaussian_copula(u1[i], u2[i], rho)
    time_cython = (time.time() - t0) / 5
    
    print(f"   Python+scipy: {time_python*1000:.1f} ms")
    print(f"   Cython:       {time_cython*1000:.1f} ms")
    print(f"   高速化: {time_python/time_cython:.1f}x")
    
    # 4. 加重相関行列
    print("\n4. 加重相関行列 (n=3000, p=20)")
    Z = np.random.randn(3000, 20).astype(np.float64)
    W = np.random.uniform(0, 1, 3000).astype(np.float64)
    
    t0 = time.time()
    for _ in range(10):
        R = compute_weighted_corr(Z, W)
    time_cython = (time.time() - t0) / 10
    print(f"   Cython: {time_cython*1000:.2f} ms")
    
    # 5. E-step バッチ処理
    print("\n5. E-step バッチ処理 (n=1000, p=10, K=3)")
    n_test = 1000
    p_test = 10
    K_test = 3
    X_cont = np.random.randn(n_test, p_test).astype(np.float64)
    X_cat_idx = np.random.randint(-1, 3, size=(n_test, 2)).astype(np.int32)
    mu = np.random.randn(K_test, p_test).astype(np.float64)
    sig = np.abs(np.random.randn(K_test, p_test)).astype(np.float64) + 0.1
    cat_probs = [np.random.dirichlet(np.ones(3), size=K_test) for _ in range(2)]
    R_list = [np.eye(p_test + 2, dtype=np.float64) for _ in range(K_test)]
    pi = np.ones(K_test, dtype=np.float64) / K_test
    
    t0 = time.time()
    for _ in range(10):
        log_resp, ll, ll_samples = compute_e_step_full(
            X_cont, X_cat_idx, mu, sig, cat_probs, R_list, pi, 5.0,
            'student_t', 'abs_rho', 'pairwise'
        )
    time_cython = (time.time() - t0) / 10
    print(f"   Cython: {time_cython*1000:.2f} ms")
    
    print("\n" + "=" * 60)
    print("ベンチマーク完了")
    print("=" * 60)


if __name__ == "__main__":
    benchmark()
