"""
Test suite for pymcmm v0.2.0

Tests cover basic functionality, Cython acceleration, and edge cases.
"""

import numpy as np
import pandas as pd
import pytest
from mcmm import MCMMGaussianCopula, MCMMGaussianCopulaSpeedy, check_acceleration


class TestAcceleration:
    """Test Cython acceleration functionality."""
    
    def test_check_acceleration(self):
        """Test that check_acceleration returns valid information."""
        info = check_acceleration()
        assert isinstance(info, dict)
        assert 'available' in info
        assert 'version' in info
        assert 'functions' in info
        assert isinstance(info['available'], bool)
        assert isinstance(info['functions'], list)


class TestMCMMBasic:
    """Basic functionality tests."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = MCMMGaussianCopula(n_components=2, random_state=42)
        assert model.K == 2
        assert model.max_iter == 100
        assert model.tol == 1e-4
    
    def test_simple_fit_predict(self):
        """Test basic fit and predict on simple data."""
        # Create simple test data
        np.random.seed(42)
        n_samples = 50
        
        df = pd.DataFrame({
            'cont1': np.random.randn(n_samples),
            'cont2': np.random.randn(n_samples),
            'cat': np.random.choice(['A', 'B'], n_samples),
            'ord': np.random.choice([1, 2, 3], n_samples)
        })
        
        model = MCMMGaussianCopula(
            n_components=2,
            max_iter=20,
            random_state=42,
            verbose=0
        )
        
        model.fit(df,
                  cont_cols=['cont1', 'cont2'],
                  cat_cols=['cat'],
                  ord_cols=['ord'])
        
        # Check that model was fitted
        assert model.pi_ is not None
        assert model.loglik_ is not None
        assert model.bic_ is not None
        
        # Test prediction
        labels = model.predict(df)
        assert len(labels) == n_samples
        assert all(0 <= label < 2 for label in labels)
        
        # Test probabilities
        probs = model.predict_proba(df)
        assert probs.shape == (n_samples, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)
    
    def test_predict_proba(self):
        """Test probability prediction."""
        np.random.seed(42)
        n_samples = 30
        
        df = pd.DataFrame({
            'cont': np.random.randn(n_samples),
            'cat': np.random.choice(['X', 'Y'], n_samples)
        })
        
        model = MCMMGaussianCopula(n_components=2, max_iter=15, random_state=42)
        model.fit(df, cont_cols=['cont'], cat_cols=['cat'])
        
        probs = model.predict_proba(df)
        assert probs.shape == (n_samples, 2)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        assert np.allclose(probs.sum(axis=1), 1.0)
    
    def test_score_samples(self):
        """Test log-likelihood scoring."""
        np.random.seed(42)
        n_samples = 30
        
        df = pd.DataFrame({
            'cont': np.random.randn(n_samples)
        })
        
        model = MCMMGaussianCopula(n_components=2, max_iter=15, random_state=42)
        model.fit(df, cont_cols=['cont'])
        
        scores = model.score_samples(df)
        assert len(scores) == n_samples
        assert all(np.isfinite(scores))
    
    def test_different_marginals(self):
        """Test different marginal distribution options."""
        np.random.seed(42)
        df = pd.DataFrame({
            'cont': np.random.randn(30),
            'ord': np.random.choice([1, 2, 3], 30)
        })
        
        # Test Gaussian marginal
        model1 = MCMMGaussianCopula(
            n_components=2,
            cont_marginal='gaussian',
            max_iter=15,
            random_state=42
        )
        model1.fit(df, cont_cols=['cont'], ord_cols=['ord'])
        assert model1.loglik_ is not None
        
        # Test Student-t marginal
        model2 = MCMMGaussianCopula(
            n_components=2,
            cont_marginal='student_t',
            max_iter=15,
            random_state=42
        )
        model2.fit(df, cont_cols=['cont'], ord_cols=['ord'])
        assert model2.loglik_ is not None
    
    def test_copula_modes(self):
        """Test different copula likelihood modes."""
        np.random.seed(42)
        df = pd.DataFrame({
            'cont1': np.random.randn(30),
            'cont2': np.random.randn(30)
        })
        
        # Test full copula
        model1 = MCMMGaussianCopula(
            n_components=2,
            copula_likelihood='full',
            max_iter=15,
            random_state=42
        )
        model1.fit(df, cont_cols=['cont1', 'cont2'])
        assert model1.loglik_ is not None
        
        # Test pairwise copula
        model2 = MCMMGaussianCopula(
            n_components=2,
            copula_likelihood='pairwise',
            max_iter=15,
            random_state=42
        )
        model2.fit(df, cont_cols=['cont1', 'cont2'])
        assert model2.loglik_ is not None


class TestMCMMSpeedy:
    """Tests for Speedy mode."""
    
    def test_speedy_initialization(self):
        """Test Speedy mode initialization."""
        model = MCMMGaussianCopulaSpeedy(
            n_components=2,
            speedy_graph='mst',
            random_state=42
        )
        assert model.speedy_graph == 'mst'
        assert model.corr_subsample == 3000
    
    def test_speedy_fit(self):
        """Test Speedy mode fitting."""
        np.random.seed(42)
        n_samples = 50
        
        df = pd.DataFrame({
            'cont1': np.random.randn(n_samples),
            'cont2': np.random.randn(n_samples),
            'cat': np.random.choice(['A', 'B'], n_samples)
        })
        
        model = MCMMGaussianCopulaSpeedy(
            n_components=2,
            max_iter=15,
            random_state=42
        )
        
        model.fit(df,
                  cont_cols=['cont1', 'cont2'],
                  cat_cols=['cat'])
        
        assert model.loglik_ is not None
        labels = model.predict(df)
        assert len(labels) == n_samples


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_cluster(self):
        """Test with single cluster."""
        np.random.seed(42)
        df = pd.DataFrame({'cont': np.random.randn(20)})
        
        model = MCMMGaussianCopula(n_components=1, max_iter=10, random_state=42)
        model.fit(df, cont_cols=['cont'])
        assert model.loglik_ is not None
    
    def test_missing_values(self):
        """Test handling of missing values."""
        np.random.seed(42)
        df = pd.DataFrame({
            'cont': [1.0, 2.0, np.nan, 4.0, 5.0],
            'cat': ['A', 'B', 'A', np.nan, 'B']
        })
        
        model = MCMMGaussianCopula(n_components=2, max_iter=10, random_state=42)
        model.fit(df, cont_cols=['cont'], cat_cols=['cat'])
        assert model.loglik_ is not None
    
    def test_small_dataset(self):
        """Test with very small dataset."""
        df = pd.DataFrame({
            'cont': [1.0, 2.0, 3.0],
            'cat': ['A', 'B', 'A']
        })
        
        model = MCMMGaussianCopula(n_components=2, max_iter=5, random_state=42)
        model.fit(df, cont_cols=['cont'], cat_cols=['cat'])
        assert model.loglik_ is not None
    
    def test_unfitted_predict_error(self):
        """Test that predict raises error on unfitted model."""
        model = MCMMGaussianCopula(n_components=2)
        df = pd.DataFrame({'cont': [1.0, 2.0, 3.0]})
        
        with pytest.raises(RuntimeError):
            model.predict(df)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

