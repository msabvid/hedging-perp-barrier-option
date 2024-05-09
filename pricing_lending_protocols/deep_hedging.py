import torch
import torch.nn as nn
import logging
import os
import tqdm
import joblib
from typing import List, Dict, Tuple

from pricing_lending_protocols.nn import RNN
from pricing_lending_protocols.risk_measure import ES


class DeepHedgingBase(nn.Module):

    def __init__(
        self,
        mean_gas_fees: float,
        mean_slippage: float,
        market_generator,
        r_cD: float,
        r_bD: float,
        r_cE: float,
        r_bE: float,
        theta0: float,
        theta: float,
    ):
        super().__init__()

        self.hedge = RNN(
            rnn_in=1 + 1,  # +1 is for time coordinate
            rnn_hidden=10,
            ffn_sizes=[10, 1],
        )
        # self.hedge = Linear([1,1])
        self.v_init = nn.Parameter(torch.tensor(1000.0))
        self.mean_gas_fees = mean_gas_fees
        self.mean_slippage = mean_slippage

        # market generator
        self.market_generator = market_generator

        # interest rates
        self.r_cD = r_cD
        self.r_bD = r_bD
        self.r_cE = r_cE
        self.r_bE = r_bE

        # ltv
        self.theta0 = theta0
        self.theta = theta

        # training error
        self.training_record = []

    def forward(
        self,
        ts: torch.Tensor,
        lag: int,
        x0: torch.Tensor,
        **kwargs,
    ):
        raise NotImplementedError

    def predict(
        self,
        ts: torch.Tensor,
        lag: int,
        p0: float,
        seed: int,
        batch_size: int,
        **kwargs,
    ):
        torch.manual_seed(seed)
        x0 = torch.ones((batch_size, 1)) * p0
        v, payoff, cost = self.forward(ts, lag=lag, x0=x0, **kwargs)
        return v, payoff, cost

    def cvar(
        self,
        ts: torch.Tensor,
        lag: int,
        level: float,
        p0: float,
        batch_size: int,
        seed: int,
        **kwargs,
    ):
        v, payoff, cost = self.predict(
            ts=ts, lag=lag, p0=p0, seed=seed, batch_size=batch_size, **kwargs
        )
        cvar = 0
        for i in range(v.shape[1]):
            cvar += ES(lam=level, x=v[:, i, :] - cost[:, i, :] - payoff[:, i, :])
        return cvar / v.shape[1]

    def mse(
        self,
        ts: torch.Tensor,
        lag: int,
        p0: float,
        batch_size: int,
        seed: int,
        **kwargs,
    ):
        v, payoff, cost = self.predict(
            ts=ts, lag=lag, p0=p0, seed=seed, batch_size=batch_size, **kwargs
        )
        return torch.mean((payoff - (v - cost)) ** 2)

    def update_training_record(self, loss: torch.Tensor):
        self.training_record.append(loss.item())

    def save(self, dir_results: str):
        torch.save(
            {"weights": self.state_dict(), "training_loss": self.training_record},
            os.path.join(dir_results, "model.pt"),
        )


class DeepHedgingBarrierOption(DeepHedgingBase):
    """
    Deep hedging of a Barrier Option from the borrower point of view
    """

    def __init__(
        self,
        mean_gas_fees: float,
        mean_slippage: float,
        market_generator,
        r_cD: float,
        r_bD: float,
        r_cE: float,
        r_bE: float,
        theta0: float,
        theta: float,
    ):
        super().__init__(
            mean_gas_fees=mean_gas_fees,
            mean_slippage=mean_slippage,
            market_generator=market_generator,
            r_cD=r_cD,
            r_bD=r_bD,
            r_cE=r_cE,
            r_bE=r_bE,
            theta0=theta0,
            theta=theta,
        )
        

    def forward(
        self,
        ts: torch.Tensor,
        lag: int,
        x0: torch.Tensor,
        **kwargs,
    ):

        batch_size = x0.shape[0]
        x = self.market_generator.sdeint(ts=ts, x0=x0, **kwargs)
        x_normalised = x / x0.unsqueeze(1)

        discretisation = ts[::lag]
        discretisation_batch = discretisation.repeat(batch_size, 1).unsqueeze(2)

        # hedge
        hedge = self.hedge(discretisation_batch, x_normalised[:, ::lag, :])

        # wealth process
        v = torch.ones((batch_size, len(discretisation), 1)) * self.v_init

        # payoff process
        payoff_ = torch.ones_like(v) * x0[0, 0] * (1 - self.theta0)

        # transaction costs
        cost = torch.zeros_like(v)

        for i, t in enumerate(discretisation[:-1]):

            p = x[:, lag * i, :]
            p_new = x[:, lag * (i + 1), :]

            h = discretisation[i + 1] - discretisation[i]
            v_new = (
                (1 + self.r_cD * h) * torch.clamp(v[:, i, :] - hedge[:, i, :] * p, min=0)
                + (1 + self.r_bD * h) * torch.clamp(v[:, i, :] - hedge[:, i, :] * p, max=0)
                + (1 + self.r_cE * h) * torch.clamp(hedge[:, i, :], min=0) * p_new
                + (1 + self.r_cE * h) * torch.clamp(hedge[:, i, :], max=0) * p_new
            )

            # calculate payoff of loan position
            payoff = p_new * torch.exp(
                self.r_cE * discretisation[i + 1]
            ) - self.theta0 * x0 * torch.exp(self.r_bD * discretisation[i + 1])
            t_vec = ts[: lag * (i + 1)].repeat(batch_size, 1).unsqueeze(2)
            liquidated = torch.any(
                (
                    self.theta * x[:, : lag * (i + 1), :] * torch.exp(self.r_cE * t_vec)
                ).less_equal(
                    self.theta0 * x0.unsqueeze(1) * torch.exp(self.r_bD * t_vec)
                ),
                dim=1,
                keepdim=False,
            )
            payoff = payoff * liquidated.logical_not()
            v_new = v_new * liquidated.logical_not()
            cost_transaction = 20 # self.mean_gas_fees * p
            change_units_eth = hedge[:,i,:] if i==0 else hedge[:,i,:] - hedge[:,i-1,:]
            transaction_costs = cost[:, i, :] + cost_transaction * (
                1 - torch.exp(- change_units_eth ** 2)
            )  # I approximate indicator function by (1-e^(x^2)) in order to backpropagate
            transaction_costs = transaction_costs * liquidated.logical_not()

            # update processes
            v[:, i + 1, :] = v_new
            payoff_[:, i + 1, :] = payoff
            cost[:, i + 1, :] = transaction_costs

        return v, payoff_, cost



class DeltaHedgeBarrierOption(DeepHedgingBarrierOption):

    def __init__(
        self,
        mean_gas_fees: float,
        mean_slippage: float,
        market_generator,
        r_cD: float,
        r_bD: float,
        r_cE: float,
        r_bE: float,
        theta0: float,
        theta: float,
    ):
        super().__init__(
            mean_gas_fees=mean_gas_fees,
            mean_slippage=mean_slippage,
            market_generator=market_generator,
            r_cD=r_cD,
            r_bD=r_bD,
            r_cE=r_cE,
            r_bE=r_bE,
            theta0=theta0,
            theta=theta,
        )

        del self.hedge
        self.hedge = lambda t, x: torch.exp(self.r_cE * t)

    
    def forward(
        self,
        ts: torch.Tensor,
        lag: int,
        x0: torch.Tensor,
        **kwargs,
    ):
        del self.v_init
        self.v_init = x0[0, 0] * (1 - self.theta0)
        return super().forward(ts=ts, lag=lag, x0=x0, **kwargs)


class DeepHedgingLoanPosition(DeepHedgingBase):
    """
    Deep hedging of Loan Position from the protocol point of view
    """

    def __init__(
        self,
        mean_gas_fees: float,
        mean_slippage: float,
        market_generator,
        r_cD: float,
        r_bD: float,
        r_cE: float,
        r_bE: float,
        theta0: float,
        theta: float,
        liquidation_bonus: float,
        liquidation_fraction: float,
    ):
        super().__init__(
            mean_gas_fees=mean_gas_fees,
            mean_slippage=mean_slippage,
            market_generator=market_generator,
            r_cD=r_cD,
            r_bD=r_bD,
            r_cE=r_cE,
            r_bE=r_bE,
            theta0=theta0,
            theta=theta,
        )

        self.liquidation_bonus = liquidation_bonus
        self.liquidation_fraction = liquidation_fraction

    def forward(
        self,
        ts: torch.Tensor,
        lag: int,
        x0: torch.Tensor,
        **kwargs,
    ):

        batch_size = x0.shape[0]
        x = self.market_generator.sdeint(ts=ts, x0=x0, **kwargs)

        hedge = self.hedge(x[:, ::lag, :])
        discretisation = ts[::lag]
        v = torch.zeros((batch_size, 1))
        for i, t in enumerate(discretisation[:-1]):

            p = x[:, lag * i, :]
            p_new = x[:, lag * (i + 1), :]

            h = discretisation[i + 1] - discretisation[i]
            v_new = (
                (1 + self.r_cD * h) * torch.clamp(v - hedge[:, i, :], min=0)
                + (1 + self.r_bD * h) * torch.clamp(v - hedge[:, i, :], max=0)
                + (1 + self.r_cE * h) * torch.clamp(hedge[:, i, :] / p, min=0) * p_new
                + (1 + self.r_cE * h) * torch.clamp(hedge[:, i, :] / p, max=0) * p_new
            )
            v = v_new

        # account for transaction costs
        v = v - self.mean_gas_fees * len(discretisation[:-1])

        # calculate payoff of loan position
        payoff = torch.zeros(batch_size)
        units_debt_asset, units_collateral_asset, recovered_units_debt_asset = (
            self.loan_position(ts, x, lag)
        )

        mask_bad_debt = units_debt_asset[:, -1] > (
            units_collateral_asset[:, -1] * x[:, -1, 0]
        )
        payoff[mask_bad_debt] = (
            recovered_units_debt_asset[mask_bad_debt]
            + units_collateral_asset[mask_bad_debt, -1] * x[mask_bad_debt, -1, 0]
            - units_debt_asset[mask_bad_debt, 0]
        )
        payoff[mask_bad_debt.logical_not()] = (
            recovered_units_debt_asset[mask_bad_debt.logical_not()]
            + units_debt_asset[mask_bad_debt.logical_not(), -1]
            - units_debt_asset[mask_bad_debt.logical_not(), 0]
        )

        # hedge payoff
        return v + payoff.unsqueeze(1)

    def loan_position(
        self, ts: torch.Tensor, prices: torch.Tensor, lag: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size = prices.shape[0]
        discretisation = ts[::lag]

        units_debt_asset = (
            torch.ones((batch_size, len(discretisation)))
            * self.theta0
            * prices[:, 0, :]
        )
        units_collateral_asset = torch.ones((batch_size, len(discretisation)))

        recovered_units_debt_asset = torch.zeros(batch_size)

        for i, t in enumerate(discretisation[1:]):
            p = prices[:, lag * (i + 1), 0]
            h = discretisation[i + 1] - discretisation[i]

            # compound interest rate
            units_collateral_asset[:, i + 1] = units_collateral_asset[:, i] * torch.exp(
                self.r_cE * h
            )
            units_debt_asset[:, i + 1] = units_debt_asset[:, i] * torch.exp(
                self.r_bD * h
            )

            # positions that are liquidated
            mask_liquidated = (
                units_collateral_asset[:, i + 1] * p * self.theta
                < units_debt_asset[:, i + 1]
            )

            # check how much of the position can be liquidated
            # in terms of the available collateral
            liquidated_fraction = torch.clamp(
                units_collateral_asset[mask_liquidated, i + 1]
                * p[mask_liquidated]
                / (
                    (1 + self.liquidation_bonus)
                    * units_debt_asset[mask_liquidated, i + 1]
                ),
                max=self.liquidation_fraction,
            )

            # liquidate
            units_collateral_asset[mask_liquidated, i + 1] -= (
                liquidated_fraction
                * units_debt_asset[mask_liquidated, i + 1]
                * 1
                / p[mask_liquidated]
                * (1 + self.liquidation_bonus)
            )

            recovered_units_debt_asset[mask_liquidated] += (
                liquidated_fraction * units_debt_asset[mask_liquidated, i + 1]
            )

            units_debt_asset[mask_liquidated, i + 1] -= (
                liquidated_fraction * units_debt_asset[mask_liquidated, i + 1]
            )

        return units_debt_asset, units_collateral_asset, recovered_units_debt_asset


def solve(
    batch_size: int,
    n_epochs: int,
    ts: torch.Tensor,
    mean_gas_fees: float,
    mean_slippage: float,
    market_generator,
    r_cD: float,
    r_bD: float,
    r_cE: float,
    r_bE: float,
    theta0: float,
    theta: float,
    p0: float,
    level: float,
    deep_hedging_type,
    dir_results: str,
    **kwargs,
):

    # logging configuration
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)
    log_file = "deep_hedging.log"
    logging.basicConfig(
        filename=os.path.join(dir_results, log_file), level=logging.DEBUG, filemode="w"
    )

    deep_hedging = deep_hedging_type(
        mean_gas_fees=mean_gas_fees,
        mean_slippage=mean_slippage,
        market_generator=market_generator,
        r_cD=r_cD,
        r_bD=r_bD,
        r_cE=r_cE,
        r_bE=r_bE,
        theta0=theta0,
        theta=theta,
    )

    lag = len(ts) // 10
    
    if deep_hedging_type == DeepHedgingBarrierOption:
        # Traing Deep hedging of Barrier option

        pbar = tqdm.tqdm(total=n_epochs)
        optimizer = torch.optim.RMSprop(
            deep_hedging.parameters(),
            lr=0.1,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[8000], gamma=0.1
        )


        # training
        deep_hedging.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            loss = deep_hedging.mse(
                ts=ts,
                lag=lag,
                batch_size=batch_size,
                p0=p0,
                seed=epoch,
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

            deep_hedging.update_training_record(loss)

            pbar.update()
            logging.info("Loss = {:.4f}".format(loss.item()))
        deep_hedging.save(dir_results=dir_results)

    # evaluate
    deep_hedging.eval()
    mse = deep_hedging.mse(ts=ts, lag=lag, batch_size=15000, p0=p0, seed=10, **kwargs)
    cvar = deep_hedging.cvar(
        ts=ts, lag=lag, level=level, batch_size=15000, p0=p0, seed=10, **kwargs
    )
    print(cvar, mse, deep_hedging.v_init)
    return cvar.item(), mse.item(), deep_hedging.v_init.item(), deep_hedging


def batch_solve(
    batch_size: int,
    n_epochs: int,
    ts: torch.Tensor,
    params: List[Dict],
    n_jobs: int = -2,
    verbose: int = 10,
):
    results = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
        joblib.delayed(solve)(batch_size=batch_size, n_epochs=n_epochs, ts=ts, **p)
        for p in params
    )
    grouped_results = []
    for i, p in enumerate(params):
        market_generator = p.pop("market_generator")
        _ = p.pop("deep_hedging_type")
        _ = p.pop("dir_results")
        grouped_results.append(
            dict(params=p, cvar=results[i][0], mse=results[i][1], v_init=results[i][2])
        )
        try:
            grouped_results[-1].update(
                {"sigma": market_generator.sigma, "mu": market_generator.mu}
            )
        except:
            pass

    return grouped_results
