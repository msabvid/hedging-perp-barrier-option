import torch
import math


class Gbm:

    def __init__(self, mu: float, sigma: float):

        self.mu = mu
        self.sigma = sigma  # annualized volatility
        self.d = 1

    def sdeint(self, ts: torch.Tensor, x0: torch.Tensor):
        """
        Euler scheme to solve the SDE.

        Parameters
        ----------
        ts: torch.Tensor
            timegrid. Vector of length N
        x0: torch.Tensor
            initial value of SDE. Tensor of shape (batch_size, d)
        brownian: Optional.
            torch.tensor of shape (batch_size, N, d)
        """

        batch_size = x0.shape[0]
        device = x0.device
        brownian_increments = torch.zeros(batch_size, len(ts), self.d, device=device)
        x = torch.ones((batch_size, len(ts), 1), device=device)
        x[:, 0, :] = x0
        for idx, t in enumerate(ts[1:]):
            h = ts[idx + 1] - ts[idx]
            brownian_increments[:, idx, :] = torch.randn(
                batch_size, self.d, device=device
            ) * torch.sqrt(h)
            x[:, idx + 1, :] = torch.exp(
                (self.mu - 0.5 * self.sigma**2) * h
                + self.sigma * brownian_increments[:, idx, :]
            )
        return x.cumprod(dim=1)


class Heston:

    def __init__(
        self, r: float, q: float, kappa: float, theta: float, sigma_v: float, rho: float
    ):
        """
        dS_t = (r-q)S_t dt + S_t * sqrt(V_t) dW_t
        dV_t = kappa * (theta - V_t) dt + sigma_v * sqrt(V_t) dW^2_t

        dW^1_t* dW^2_t = rho * dt
        """

        self.r = r
        self.q = q
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho

    def sdeint(self, ts: torch.Tensor, x0: torch.Tensor, v0: float):
        """
        Euler scheme to solve the SDE.

        Parameters
        ----------
        ts: torch.Tensor
            timegrid. Vector of length N
        x0: torch.Tensor
            initial value of SDE. Tensor of shape (batch_size, d)
        brownian: Optional.
            torch.tensor of shape (batch_size, N, d)
        """

        v0 = torch.ones_like(x0) * v0
        batch_size = x0.shape[0]

        device = x0.device
        z1 = torch.randn(batch_size, len(ts) - 1, 1, device=device)
        z_ = torch.randn(batch_size, len(ts) - 1, 1, device=device)
        z2 = self.rho * z1 + math.sqrt(1 - self.rho**2) * z_

        s = torch.ones((batch_size, len(ts), 1), device=device)
        s[:, 0, :] = x0
        v = torch.ones((batch_size, len(ts), 1), device=device)
        v[:, 0, :] = v0

        for idx, t in enumerate(ts[1:]):
            h = ts[idx + 1] - ts[idx]
            dW1 = z1[:, idx, :] * torch.sqrt(h)
            dW2 = z2[:, idx, :] * torch.sqrt(h)
            s[:, idx + 1, :] = (
                s[:, idx, :]
                + self.r * s[:, idx, :] * h
                + torch.sqrt(v[:, idx, :]) * s[:, idx, :] * dW1
            )
            v[:, idx + 1, :] = (
                v[:, idx, :]
                + self.kappa * (self.theta - v[:, idx, :]) * h
                + torch.sqrt(v[:, idx, :]) * self.sigma_v * dW2
            )

        return s
