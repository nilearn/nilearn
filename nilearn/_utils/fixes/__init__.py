try:
    from sklearn.metrics import check_scoring
except ImportError:
    # scikit-learn < 0.20
    from .sklearn_metrics import check_scoring

__all__ = ['check_scoring']
