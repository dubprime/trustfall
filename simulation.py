from dataclasses import dataclass
from typing import Dict, List
import numpy as np

# Constants
COLORS: Dict[int, str] = {1: "#e45756", 2: "#4c78a8", 3: "#f58518"}
TERMS = [1, 2, 3]

@dataclass
class InputParams:
    initial_capital: float
    monthly_interest: float  # as decimal (e.g. 0.12)
    num_months: int
    default_rates: Dict[int, float]
    allocation: Dict[int, float]
    show_def_overlay: bool

@dataclass
class SimulationResults:
    months: np.ndarray
    new_by_t: np.ndarray
    payment_schedule: np.ndarray
    out_gross: Dict[int, np.ndarray]
    out_net: Dict[int, np.ndarray]
    def_out: Dict[int, np.ndarray]
    defaults: List[float]
    interest: List[float]
    reinvest: List[float]
    payment_by_tenor: Dict[int, np.ndarray]
    interest_by_tenor: Dict[int, np.ndarray]
    principal_by_tenor: Dict[int, np.ndarray]
    default_by_tenor: Dict[int, np.ndarray]
    pmt_factors: Dict[int, float]

def run_simulation(params: InputParams) -> SimulationResults:
    """
    Perform the core portfolio simulation: allocate capital, apply defaults,
    calculate amortization, and collect cashflows by tenor.
    """
    r = params.monthly_interest
    d = params.default_rates
    p = params.allocation
    T = params.num_months

    # Initialize schedules
    payment_schedule = np.zeros(T + max(TERMS) + 1)
    payment_schedule[0] = params.initial_capital
    new_by_t = np.zeros((T, len(TERMS)))

    # Trackers per tenor over the full schedule horizon
    payment_by_tenor = {n: np.zeros_like(payment_schedule) for n in TERMS}
    interest_by_tenor = {n: np.zeros_like(payment_schedule) for n in TERMS}
    principal_by_tenor = {n: np.zeros_like(payment_schedule) for n in TERMS}
    default_by_tenor = {n: np.zeros_like(payment_schedule) for n in TERMS}

    # Precompute level-payment factors and per-period hazards
    pmt_factors = {n: r / (1 - (1 + r) ** (-n)) for n in TERMS if n > 1}
    hazard = {n: 1 - (1 - d[n]) ** (1 / n) for n in TERMS if n > 1}

    defaults, interest, reinvest = [], [], []

    # Monthly simulation loop
    for t in range(T):
        available_cash = payment_schedule[t]
        deployed = np.array([p[n] * available_cash for n in TERMS])
        new_by_t[t, :] = deployed

        month_loss = month_int = month_reinv = 0.0

        # 1-month bullet loan
        P1 = deployed[0]
        if P1 > 0:
            loss1 = d[1] * P1
            default_by_tenor[1][t] += loss1
            recovered = P1 - loss1
            int1 = r * recovered
            idx = t + 1
            payment_schedule[idx] += recovered + int1
            payment_by_tenor[1][idx] += recovered + int1
            interest_by_tenor[1][idx] += int1
            principal_by_tenor[1][idx] += recovered
            month_loss += loss1
            month_int += int1
            month_reinv += (recovered + int1)

        # 2- & 3-month amortizing loans
        for i, n in enumerate(TERMS[1:], start=1):
            Pn = deployed[i]
            if Pn <= 0:
                continue
            pf = pmt_factors[n]
            hz = hazard[n]
            remaining = Pn
            pmt = pf * Pn
            for k in range(1, n + 1):
                loss_k = hz * remaining
                remaining -= loss_k
                default_by_tenor[n][t + k] += loss_k

                int_k = r * remaining
                prin_k = pmt - int_k
                remaining -= prin_k

                idx2 = t + k
                payment_schedule[idx2] += prin_k + int_k
                payment_by_tenor[n][idx2] += prin_k + int_k
                interest_by_tenor[n][idx2] += int_k
                principal_by_tenor[n][idx2] += prin_k

                month_loss += loss_k
                month_int += int_k
                month_reinv += (prin_k + int_k)

        defaults.append(month_loss)
        interest.append(month_int)
        reinvest.append(month_reinv)

    months = np.arange(T)

    # Build outstanding (gross & net) and default schedules
    weights_gross = {1: [1, 0]}
    weights_net = {1: [1 - d[1], 0]}
    for n in TERMS[1:]:
        wg = [(1 + r) ** k - (((1 + r) ** k - 1) * pmt_factors[n] / r) for k in range(n + 1)]
        wn = [wg[k] * (1 - hazard[n]) ** k for k in range(n + 1)]
        weights_gross[n], weights_net[n] = wg, wn

    out_gross, out_net, def_out = {}, {}, {}
    for idx, n in enumerate(TERMS):
        out_gross[n] = np.array([
            sum(new_by_t[j, idx] * weights_gross[n][min(t - j, n)] for j in range(t + 1))
            for t in months
        ])
        out_net[n] = np.array([
            sum(new_by_t[j, idx] * weights_net[n][min(t - j, n)] for j in range(t + 1))
            for t in months
        ])
        def_out[n] = out_gross[n] - out_net[n]

    return SimulationResults(
        months=months,
        new_by_t=new_by_t,
        payment_schedule=payment_schedule,
        out_gross=out_gross,
        out_net=out_net,
        def_out=def_out,
        defaults=defaults,
        interest=interest,
        reinvest=reinvest,
        payment_by_tenor=payment_by_tenor,
        interest_by_tenor=interest_by_tenor,
        principal_by_tenor=principal_by_tenor,
        default_by_tenor=default_by_tenor,
        pmt_factors=pmt_factors,
    ) 