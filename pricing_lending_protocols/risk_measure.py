import torch


def VaR(alpha: float, x: torch.Tensor):
    """
    VaR_alpha(X) = -F_X^{-1}(alpha)
    """
    percentile = torch.quantile(x, q=alpha)
    return -percentile


def ES(lam: float, x: torch.Tensor):
    """
    Expected Shortfall
    """
    empirical_var = VaR(alpha=lam, x=x)
    mask = x[x < -empirical_var]

    N = len(x)

    expected_shortfall = (
        -1 / lam * (1 / N * mask.sum() + empirical_var * (1 / N * len(mask) - lam))
    )
    return expected_shortfall
