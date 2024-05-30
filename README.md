# Hedging perpetual down-and-out barrier option


## Installation & Running

This repo uses [hatch](https://hatch.pypa.io/latest/) for dependency
management. In order to run the examples, first activate the environment 

```
hatch shell
```

and then run `python deep_hedging.py` for one combination of parameters (initial loan to value, market drift, market vol),
or `python batch_deep_hedging.py` for parallel deep hedging for different combinations of parameters.

Alternatively, you can directly run 
```
hatch run examples:deep_hedging
```

Results of `python batch_deep_hedging.py` include
- CVaR of the hedging strategy at terminal time T
- MSE of the hedging strategy
- Initial capital needed to hedge the option

and are saved in the  `results/deep_hedging.json`.

Model weights for the hedging stragegy are saved in the file `results/mu{}_sigma{}_theta{}/model.pt`


## Context

 We first analyse lending contract from the borrower perspective.
 To open a long-ETH loan position, an agent
 1. purchases 1 ETH on the market for $P_0$ of USDC,
 2. deposits 1 ETH as collateral,
 3. borrow $\theta \,P_0$ USDC against the collateral.

 We see that effectively only $P_0(1 - \theta^0)$ is required to establish the position. For capital efficiency the agent may use a flashswap (or a flashloan and a swap) along the following steps:
1. Begin with $P_0(1-\theta^0)$ of USDC.
2. Obtain 1 ETH using flashswap (need to deposit $P_0$ USDC within one block for this to materialise)\footnote{One can also get USDC via flashloan and swap it for ETH, but this involves additional gas fee, fee for trading and suffer from temporal market impact}.
3. Deposit 1 ETH as collateral and start earning interests according to $e^{r^{c,E}\,t}$. If there is no rehypothecation of collateral, $r^{c,E}=0$.
4. Borrow $\theta^0 \,P_0$ of USDC against the collateral and start paying interests according to $\theta^0 \,P_0 e^{r^{b,D}\, t}$
5. Put together $\theta^0 \,P_0$ and initial amount $P_0(1-\theta^0)$  of USDC to complete the flashswap.


The loan position has a maturity $T>0$.
At any time $t\in [0,T]$, the holder of the position may choose to pay back the loan $\theta^0 \,P_0 e^{r^{b,D}\, t}$ in exchange for the collateral with value $P_t\,e^{r^{c,E}\,t} $. Note that a rational agent will only do that if $\theta^0 \,P_0 e^{r^{b,D}\, t} \leq P_t\,e^{r^{c,E}\,t}$, otherwise it is better to walk away from the position.  Hence,   the agent is entitled to the  payoff
$$(P_t\,e^{r^{c,E}\,t} -\theta^0 \,P_0 e^{r^{b,D}\, t} )^{+},$$
 where $x^+ = \max\{0,x\}$.

 Many leading protocols have liquidation constraints.
 If the value of the asset falls too low, the position will be liquidated.
 Let $\theta \in (\theta^0,1)$ be a liquidation threshold (LLTV),
 and let $\tau^B$ be the liquidation time  defined by

$$\tau^B := \inf \left\\{ t \in [0,T] \mid \theta P_t e^{r^{c,E} \, t} \leq \theta^0 \,P_0\ e^{r^{b,D} \,t}  \right\\}\,.$$

Since LLTV $\theta<1$ then for all $t<\tau^B$

$$0< \theta P_t e^{r^{c,E} t} - \theta^0 P_0 e^{r^{b,D}} < P_t e^{r^{c,E} t} -  \theta^0 P_0\ e^{r^{b,D}}\,.
$$

The payoff accounting for liquidations is given by
$$\psi(t, P_t) = (P_t\,e^{r^{c,E}\,t} - \theta^0 \,P_0\,e^{r^{b,D}\, t} )\mathbf{1}_{\{t<\tau^B\}}\,.$$

Note that this contract is  equivalent to a down-and-out barrier option, where the position becomes worthless to its holder when the value of the collateral falls sufficiently low.
