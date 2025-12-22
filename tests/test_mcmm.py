
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t as student_t

from mcmm import MCMMGaussianCopula, MCMMGaussianCopulaSpeedy


def _check_cython_available():
    
    try:
        from mcmm._fast_core import py_norm_cdf
        return True
    except ImportError:
        return False



@pytest.fixture
def simple_data():
    
    np.random.seed(42)
    n = 200
    
    data = {
        'cont1': np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100)]),
        'cont2': np.concatenate([np.random.normal(10, 2, 100), np.random.normal(5, 2, 100)]),
        'cat1': np.concatenate([
            np.random.choice(['A', 'B'], 100, p=[0.8, 0.2]),
            np.random.choice(['A', 'B'], 100, p=[0.2, 0.8])
        ]),
        'ord1': np.concatenate([
            np.random.choice([1, 2, 3], 100, p=[0.6, 0.3, 0.1]),
            np.random.choice([1, 2, 3], 100, p=[0.1, 0.3, 0.6])
        ])
    }
    return pd.DataFrame(data)


@pytest.fixture
def data_with_missing():
    
    np.random.seed(42)
    n = 150
    
    df = pd.DataFrame({
        'cont1': np.random.randn(n),
        'cont2': np.random.randn(n),
        'cat1': np.random.choice(['X', 'Y', 'Z'], n),
        'ord1': np.random.choice([1, 2, 3, 4], n),
    })
    
    df.loc[np.random.choice(n, 20, replace=False), 'cont1'] = np.nan
    df.loc[np.random.choice(n, 15, replace=False), 'cat1'] = None
    
    return df



class TestMCMMGaussianCopula:
    
    
    def test_init(self):
        
        model = MCMMGaussianCopula(n_components=3)
        assert model.K == 3
        assert model.cont_marginal == 'student_t'
        assert model.copula_likelihood == 'full'
    
    def test_fit_basic(self, simple_data):
        
        model = MCMMGaussianCopula(
            n_components=2,
            max_iter=10,
            random_state=42,
            verbose=0
        )
        model.fit(
            simple_data,
            cont_cols=['cont1', 'cont2'],
            cat_cols=['cat1'],
            ord_cols=['ord1']
        )
        
        assert model.pi_ is not None
        assert len(model.pi_) == 2
        assert np.allclose(model.pi_.sum(), 1.0)
        assert model.loglik_ is not None
        assert model.bic_ is not None
    
    def test_predict(self, simple_data):
        
        model = MCMMGaussianCopula(n_components=2, max_iter=10, random_state=42)
        model.fit(simple_data, cont_cols=['cont1', 'cont2'], cat_cols=['cat1'], ord_cols=['ord1'])
        
        clusters = model.predict(simple_data)
        assert len(clusters) == len(simple_data)
        assert set(clusters).issubset({0, 1})
    
    def test_predict_proba(self, simple_data):
        
        model = MCMMGaussianCopula(n_components=2, max_iter=10, random_state=42)
        model.fit(simple_data, cont_cols=['cont1', 'cont2'], cat_cols=['cat1'], ord_cols=['ord1'])
        
        proba = model.predict_proba(simple_data)
        assert proba.shape == (len(simple_data), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_missing_values(self, data_with_missing):
        
        model = MCMMGaussianCopula(n_components=2, max_iter=10, random_state=42)
        model.fit(
            data_with_missing,
            cont_cols=['cont1', 'cont2'],
            cat_cols=['cat1'],
            ord_cols=['ord1']
        )
        
        clusters = model.predict(data_with_missing)
        assert len(clusters) == len(data_with_missing)


class TestMCMMGaussianCopulaSpeedy:
    
    
    def test_init(self):
        
        model = MCMMGaussianCopulaSpeedy(
            n_components=3,
            speedy_graph='mst',
            corr_subsample=1000
        )
        assert model.K == 3
        assert model.speedy_graph == 'mst'
        assert model.corr_subsample == 1000
    
    def test_fit_mst(self, simple_data):
        
        model = MCMMGaussianCopulaSpeedy(
            n_components=2,
            speedy_graph='mst',
            max_iter=10,
            random_state=42
        )
        model.fit(simple_data, cont_cols=['cont1', 'cont2'], cat_cols=['cat1'], ord_cols=['ord1'])
        
        assert model.speedy_edges_ is not None
        assert len(model.speedy_edges_) == 2
    
    def test_fit_knn(self, simple_data):
        
        model = MCMMGaussianCopulaSpeedy(
            n_components=2,
            speedy_graph='knn',
            speedy_k_per_node=2,
            max_iter=10,
            random_state=42
        )
        model.fit(simple_data, cont_cols=['cont1', 'cont2'], cat_cols=['cat1'], ord_cols=['ord1'])
        
        assert model.speedy_edges_ is not None


class TestCythonFunctions:
    
    
    def test_cython_import(self):
        
        try:
            from mcmm._fast_core import py_norm_cdf
            cython_available = True
        except ImportError:
            cython_available = False
        
        print(f"Cython available: {cython_available}")
    
    @pytest.mark.skipif(
        not _check_cython_available(),
        reason="Cython not compiled"
    )
    def test_norm_cdf_accuracy(self):
        
        from mcmm._fast_core import py_norm_cdf
        
        x_test = np.linspace(-5, 5, 100)
        scipy_result = norm.cdf(x_test)
        cython_result = np.array([py_norm_cdf(xi) for xi in x_test])
        
        max_error = np.max(np.abs(scipy_result - cython_result))
        assert max_error < 1e-6, f"Max error: {max_error}"
    
    @pytest.mark.skipif(
        not _check_cython_available(),
        reason="Cython not compiled"
    )
    def test_studentt_cdf_accuracy(self):
        
        from mcmm._fast_core import py_studentt_cdf
        
        x_test = np.linspace(-5, 5, 100)
        for nu in [2.5, 5, 10, 30]:
            scipy_result = student_t.cdf(x_test, df=nu)
            cython_result = np.array([py_studentt_cdf(xi, nu) for xi in x_test])
            
            max_error = np.max(np.abs(scipy_result - cython_result))
            assert max_error < 1e-4, f"Max error for nu={nu}: {max_error}"



class TestIntegration:
    
    
    def test_reproducibility(self, simple_data):
        
        model1 = MCMMGaussianCopulaSpeedy(n_components=2, max_iter=10, random_state=123)
        model1.fit(simple_data, cont_cols=['cont1', 'cont2'], cat_cols=['cat1'], ord_cols=['ord1'])
        
        model2 = MCMMGaussianCopulaSpeedy(n_components=2, max_iter=10, random_state=123)
        model2.fit(simple_data, cont_cols=['cont1', 'cont2'], cat_cols=['cat1'], ord_cols=['ord1'])
        
        assert np.allclose(model1.loglik_, model2.loglik_)
        assert np.allclose(model1.pi_, model2.pi_)
    
    def test_different_marginals(self, simple_data):
        
        for marginal in ['gaussian', 'student_t']:
            model = MCMMGaussianCopula(
                n_components=2,
                cont_marginal=marginal,
                max_iter=10,
                random_state=42
            )
            model.fit(simple_data, cont_cols=['cont1', 'cont2'], cat_cols=['cat1'], ord_cols=['ord1'])
            
            assert model.loglik_ is not None
    
    def test_outlier_detection(self, simple_data):
        
        model = MCMMGaussianCopula(n_components=2, max_iter=10, random_state=42)
        model.fit(simple_data, cont_cols=['cont1', 'cont2'], cat_cols=['cat1'], ord_cols=['ord1'])
        
        is_outlier, scores, threshold = model.detect_outliers(simple_data, q=5.0)
        
        assert len(is_outlier) == len(simple_data)
        assert len(scores) == len(simple_data)
        assert isinstance(threshold, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
