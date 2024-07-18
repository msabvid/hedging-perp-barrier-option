import argparse
import json
from itertools import product

import torch

from pricing_lending_protocols import deep_hedging
from pricing_lending_protocols.market_generator import Gbm

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="Deep Hedging")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--n_epochs", type=int, default=10000)
    args = parser.parse_args()

    # time discretisation
    n_steps = 50
    ts = torch.linspace(0, 0.2, n_steps + 1)

    # interest rates. https://app.aave.com/markets/
    r_bD = 0.12
    r_cD = 0.08
    r_bE = 0.025
    r_cE = 0.017

    # gas fees
    mean_gas_fees = 20  # 5e-9  # https://etherscan.io/gastracker, 5gwei = 5 * 10^-9 ETH
    # units_gas_transaction = 100000  # 21000
    # mean_gas_fees = mean_gas_fees * units_gas_transaction

    # random seed
    torch.random.seed = 1

    # market parameters
    sigmas = [0.1, 0.3, 0.5]
    mus = [-0.3, 0.0, 0.3]

    # inital theta^0
    initial_ltvs = [0.81, 0.83, 0.85, 0.87, 0.89]

    # loan-to-values
    theta = 0.9

    # parameters Deep Hedging
    params = [
        dict(
            mean_gas_fees=mean_gas_fees,
            mean_slippage=0,
            market_generator=Gbm(mu, sigma),
            r_cD=r_cD,
            r_bD=r_bD,
            r_cE=r_cE,
            r_bE=r_bE,
            theta0=theta0,
            theta=theta,
            p0=2000,
            level=0.1,
            deep_hedging_type=deep_hedging.DeepHedgingBarrierOption,
            dir_results=f"results/mu{mu}_sigma{sigma}_theta0{theta0}",
        )
        for mu, sigma, theta0 in product(mus, sigmas, initial_ltvs)
    ]

    grouped_results = deep_hedging.batch_solve(
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        ts=ts,
        params=params,
    )
    with open("results/deep_hedging.json", "w") as f:
        json.dump(grouped_results, f, indent=4)

    # parameters Delta hedge
    params = [
        dict(
            mean_gas_fees=mean_gas_fees,
            mean_slippage=0,
            market_generator=Gbm(mu, sigma),
            r_cD=r_cD,
            r_bD=r_bD,
            r_cE=r_cE,
            r_bE=r_bE,
            theta0=theta0,
            theta=theta,
            p0=2000,
            level=0.1,
            deep_hedging_type=deep_hedging.DeltaHedgeBarrierOption,
            dir_results=f"results/mu{mu}_sigma{sigma}_theta0{theta0}",
        )
        for mu, sigma, theta0 in product(mus, sigmas, initial_ltvs)
    ]

    grouped_results = deep_hedging.batch_solve(
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        ts=ts,
        params=params,
    )
    with open("results/delta_hedging.json", "w") as f:
        json.dump(grouped_results, f, indent=4)
