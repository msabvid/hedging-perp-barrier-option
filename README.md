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
$$\psi(t, P_t) = (P_t\,e^{r^{c,E}\,t} - \theta^0 \,P_0\,e^{r^{b,D}\, t} )^+\mathbf{1}_{\{t<\tau^B\}}\,.$$

Note that this contract is  equivalent to a down-and-out barrier option, where the position becomes worthless to its holder when the value of the collateral falls sufficiently low.

## Non-linear pricing framework
Let $(V_t)\_{t\in [0,T]}$ be the wealth process   (in USDC) and
$(\pi_t)_{t \in [0,T]}$
be the   portfolio process representing the number of units invested in ETH.

Let $\mathcal T$ be the set of stopping times taking values in $[0,T]$.
Let  $\tau \in \mathcal T$ be a stopping time
at which  the holder chooses to pay back the loan. The agent's   payoff at
$\tau$ is then given by
$$\psi(\tau, P_\tau) = (P_\tau\,e^{r^{c,E}\,\tau} - \theta^0 \,P_0\,e^{r^{b,D}\, \tau} )^+\mathbf{1}_{\{\tau<\tau^B\}}\,.$$

For any $t<\tau$, given $V_{t}$ and $\pi_t$,
the value of wealth at ${t+\Delta t }$ for a small $\Delta t$ is given by

$$
\begin{split}
     V_{t+\Delta t} &= (1 + r^{c,D} \Delta t)(V_t - \pi_tP_t)^+ - (1 + r^{b,D} \Delta t)(V_t - \pi_t P_t)^- \\
    &\qquad + (1 + r^{c,E} \Delta t) \left(\pi_t \right)^+ (P_t + \Delta P_t) - (1 + r^{b,E} \Delta t) \left(\pi_t \right)^- (P_t + \Delta P_t)\,.
\end{split}
$$

The first term is  due to  the interest rate earned by providing/holding USDC collateral,
the second term is  due to the interest rate paid for  borrowing USDC collateral,
the third term is   the value of wealth  at $t+\Delta t$ due to holding   ETH,
and the last term is the cost due to shortselling ETH from an external market.

We define the total cost of the trading strategy $\pi$ as

$$
C_{t_n}(\pi) := \sum_{k=0}^n c_k(\pi_{t_k} - \pi_{t_{k-1}}),
$$

where $c_k : \mathbb R \rightarrow \mathbb R_+$ is some non-negative function.

Next, we parametrise $\pi_t$ by a recurrent neural network such that 

$$\pi_{t_k} \approx \pi^{\phi^*}(P_{t_k}, \pi_{t_k})$$ 

with $\phi^* \in \mathbb R^p, v_0^*\in \mathbb R$ the network's parameters and the initial wealth value satisfying

$$
(\phi^\*, v_0^\*) := \arg \inf_{\phi, v_0} \quad \sum_{k=1}^n \mathbb E\left\[\psi(t_k, P_{t_k}) - (V_{t_k} - C_{t_k}(\pi^\phi)) \right\]^2.
$$

The above optimisation obtains the initial wealth $v_0^\*$ and the hedging strategy $\pi^{\phi^\*}$ such that at every time step $t_k$ the wealth process $V_{t_k}$ hedges the payoff of the Barrier option, accounting for transaction costs, *regardless of the exercise time of the option*.
