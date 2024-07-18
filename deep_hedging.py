import argparse

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
    mean_gas_fees = 20  # https://etherscan.io/gastracker, 5gwei = 5 * 10^-9 ETH
    # units_gas_transaction = 100000  # 21000
    # mean_gas_fees = mean_gas_fees * units_gas_transaction

    # random seed
    torch.random.seed = 1

    # market generator
    market_generator = Gbm(mu=0.1, sigma=0.5)

    # loan-to-values
    theta = 0.9
    theta0 = 0.85

    cvar, _, _, _ = deep_hedging.solve(
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        ts=ts,
        mean_gas_fees=mean_gas_fees,
        mean_slippage=0,
        market_generator=market_generator,
        r_cD=r_cD,
        r_bD=r_bD,
        r_cE=r_cE,
        r_bE=r_bE,
        theta0=theta0,
        theta=theta,
        p0=2000,
        level=0.1,
        deep_hedging_type=deep_hedging.DeepHedgingBarrierOption,  # deep_hedging_type
        # deep_hedging_type=deep_hedging.DeltaHedgeBarrierOption,  # deep_hedging_type
        dir_results="results",
    )
    print(cvar)
