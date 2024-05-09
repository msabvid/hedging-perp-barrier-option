import numpy as np
from scipy import integrate, optimize

# Reference: Lipton 2002 - the vol smile problem


def phi(v0, kappa, theta, sigma_v, rho, u, tau):
    alpha_hat = -0.5 * u * (u + 1j)
    beta = kappa - 1j * u * sigma_v * rho
    gamma = 0.5 * sigma_v**2
    d = np.sqrt(beta**2 - 4 * alpha_hat * gamma)
    g = (beta - d) / (beta + d)
    h = np.exp(-d * tau)
    A_ = (beta - d) * tau - 2 * np.log((g * h - 1) / (g - 1))
    A = kappa * theta / (sigma_v**2) * A_
    B = (beta - d) / (sigma_v**2) * (1 - h) / (1 - g * h)
    return np.exp(A + B * v0)


def integral(v0, kappa, theta, sigma_v, rho, k, tau):
    def integrand(u):
        return np.real(
            np.exp((1j * u + 0.5) * k)
            * phi(v0, kappa, theta, sigma_v, rho, u - 0.5j, tau)
        ) / (u**2 + 0.25)

    i, err = integrate.quad_vec(integrand, 0, 1000)
    return i


def call(r, q, s0, v0, kappa, theta, sigma_v, rho, k, tau):
    a = np.log(s0 / k) + (r - q) * tau
    i = integral(v0, kappa, theta, sigma_v, rho, a, tau)
    return s0 * np.exp(-q * tau) - k * np.exp(-r * tau) / np.pi * i


def calibrate(
    s0: float,
    v0: float,
    r: float,
    q: float,
    tau: float,
    call_prices: np.ndarray[float],
    strikes: np.ndarray[float],
):
    def mse(x):
        kappa = x[0]
        theta = x[1]
        sigma_v = x[2]
        rho = x[3]
        price = call(
            r=r,
            q=q,
            s0=s0,
            v0=v0,
            kappa=kappa,
            theta=theta,
            sigma_v=sigma_v,
            rho=rho,
            k=strikes,
            tau=tau,
        )
        return np.mean((price - call_prices) ** 2)

    x0 = np.array([1.0, 0.04, 0.2, 0.1])
    # constraints for rho \in [-1,1]
    cons = (
        {"type": "ineq", "fun": lambda x: x[3] + 1},
        {"type": "ineq", "fun": lambda x: 1 - x[3]},
    )
    res = optimize.minimize(mse, x0=x0, constraints=cons)
    return res
