#!/usr/bin/env python3
# mm1_sim.py
# Discrete-event M/M/1 simulator (no external sim libs). Plots with matplotlib.

import math, random, csv, argparse
from statistics import mean, stdev

try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    np = None
    plt = None


# ---------- Core simulator ----------
def mm1_custom(lambda_rate=0.8, mu_rate=1.0, sim_time=30000.0, warmup=3000.0, seed=123):
    """
    M/M/1 simulation with exponential interarrival (lambda_rate) and service (mu_rate).
    Returns dict of metrics. Uses a tiny custom event loop (arrivals, departures, queue samples).
    """
    rng = random.Random(seed)
    t = 0.0
    server_busy_until = 0.0
    queue = []  # store arrival times for waiting customers

    arrivals, starts, departs = [], [], []
    q_time, q_len = [], []

    def exp(rate):  # exponential RV
        return rng.expovariate(rate)

    next_arrival = exp(lambda_rate)
    next_departure = math.inf
    next_q_sample = 0.0  # sample queue length every 1.0

    while t < sim_time:
        t_next = min(next_arrival, next_departure, next_q_sample)
        if t_next == math.inf:
            break
        t = t_next

        if t == next_arrival:
            # arrival
            arrivals.append(t)
            if t >= server_busy_until and len(queue) == 0:
                # starts service immediately
                service = exp(mu_rate)
                starts.append(t)
                server_busy_until = t + service
                next_departure = server_busy_until
            else:
                # join queue (FIFO)
                queue.append(t)
            next_arrival = t + exp(lambda_rate)

        elif t == next_departure:
            # departure
            departs.append(t)
            if queue:
                _a = queue.pop(0)
                s = t
                starts.append(s)
                service = exp(mu_rate)
                server_busy_until = t + service
                next_departure = server_busy_until
            else:
                server_busy_until = t
                next_departure = math.inf

        else:
            # queue sample
            q_time.append(t)
            q_len.append(len(queue))
            next_q_sample = t + 1.0

        if next_q_sample == 0.0 and t > 0.0:
            next_q_sample = t + 1.0

    # align arrays (only departed customers count for Wq/W)
    n_done = len(departs)
    arrivals = arrivals[:n_done]
    starts = starts[:n_done]

    # filter stats after warmup
    waits, sys_times = [], []
    for i in range(n_done):
        if departs[i] >= warmup:
            waits.append(starts[i] - arrivals[i])
            sys_times.append(departs[i] - arrivals[i])

    if len(waits) == 0:
        avg_wait = float("nan")
        avg_system = float("nan")
        throughput = 0.0
    else:
        avg_wait = sum(waits) / len(waits)
        avg_system = sum(sys_times) / len(sys_times)
        throughput = len(waits) / max(1.0, (sim_time - warmup))

    # average queue length (time-average via samples after warmup)
    avg_q = float("nan")
    if q_time:
        vals = [q_len[i] for i in range(len(q_time)) if q_time[i] >= warmup]
        if vals:
            avg_q = sum(vals) / len(vals)

    rho = lambda_rate / mu_rate if mu_rate > 0 else float("inf")
    return {
        "lambda": lambda_rate,
        "mu": mu_rate,
        "rho": rho,
        "avg_wait": avg_wait,        # Wq
        "avg_system": avg_system,    # W
        "avg_queue_len": avg_q,      # Lq
        "throughput": throughput,
        "completed_customers": len(waits),
    }


# ---------- Helpers ----------
def sweep_lambdas(mu, n_points=10, max_rho=0.95):
    """Evenly spaced lambdas from 0.1 to max_rho*mu."""
    lambdas = []
    start = 0.1
    stop = max_rho * mu
    step = (stop - start) / (n_points - 1)
    for i in range(n_points):
        lambdas.append(start + i * step)
    return lambdas


def write_csv(path, rows, header):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def plot_lines(x, y, title, xlab, ylab, out_png, theory_x=None, theory_y=None, yerr=None, legend_labels=None):
    if plt is None or np is None:
        print("[WARN] matplotlib/numpy not installed — skipping plot:", out_png)
        return
    fig = plt.figure()
    ax = fig.gca()
    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, label=(legend_labels[0] if legend_labels else "Simulation"))
    else:
        ax.plot(x, y, marker="o", label=(legend_labels[0] if legend_labels else "Simulation"))
    if theory_x is not None and theory_y is not None:
        ax.plot(theory_x, theory_y, label=(legend_labels[1] if legend_labels else "Theory"))
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print("Saved plot:", out_png)


# ---------- Main workflows ----------
def run_single(mu=1.0, sim_time=30000, warmup=3000, n_points=10, seed=123, out_prefix="mm1_single"):
    lambdas = sweep_lambdas(mu, n_points=n_points, max_rho=0.95)
    rows = []
    for i, lam in enumerate(lambdas):
        res = mm1_custom(lam, mu, sim_time, warmup, seed + i)
        rows.append([res["lambda"], res["mu"], res["rho"], res["avg_wait"], res["avg_queue_len"], res["throughput"]])
    csv_path = f"{out_prefix}.csv"
    write_csv(csv_path, rows, ["lambda", "mu", "rho", "avg_wait", "avg_queue_len", "throughput"])
    print("Saved CSV:", csv_path)

    # plots
    if np is not None and plt is not None:
        import numpy as _np
        data = _np.array(rows, dtype=float)
        rho = data[:, 2]
        wq = data[:, 3]
        lq = data[:, 4]
        plot_lines(rho, wq, "M/M/1: Waiting Time vs Utilization", "Utilization (rho)", "Average Waiting Time (Wq)", f"{out_prefix}_wait.png")
        plot_lines(rho, lq, "M/M/1: Queue Length vs Utilization", "Utilization (rho)", "Average Queue Length (Lq)", f"{out_prefix}_queue.png")


def run_multi(mu=1.0, sim_time=30000, warmup=3000, n_points=12, seeds=20, seed0=2025, out_prefix="mm1_multi"):
    lambdas = sweep_lambdas(mu, n_points=n_points, max_rho=0.95)
    rows = []
    for lam in lambdas:
        wq_vals, lq_vals = [], []
        for i in range(seeds):
            res = mm1_custom(lam, mu, sim_time, warmup, seed0 + i)
            wq_vals.append(res["avg_wait"])
            lq_vals.append(res["avg_queue_len"])
        rho = lam / mu
        wq_m = mean(wq_vals)
        lq_m = mean(lq_vals)
        wq_s = stdev(wq_vals) if len(wq_vals) > 1 else 0.0
        lq_s = stdev(lq_vals) if len(lq_vals) > 1 else 0.0
        # 95% CI ~ mean ± 1.96 * (s / sqrt(n))
        ci_wq = 1.96 * (wq_s / (seeds ** 0.5)) if seeds > 1 else 0.0
        ci_lq = 1.96 * (lq_s / (seeds ** 0.5)) if seeds > 1 else 0.0
        rows.append([lam, mu, rho, wq_m, ci_wq, lq_m, ci_lq, seeds])

    csv_path = f"{out_prefix}.csv"
    write_csv(csv_path, rows, ["lambda", "mu", "rho", "wait_mean", "wait_ci95", "q_mean", "q_ci95", "seeds"])
    print("Saved CSV:", csv_path)

    # plots with theory overlays
    if np is not None and plt is not None:
        import numpy as _np
        data = _np.array(rows, dtype=float)
        rho = data[:, 2]
        wq_m, wq_ci = data[:, 3], data[:, 4]
        lq_m, lq_ci = data[:, 5], data[:, 6]

        rho_dense = _np.linspace(0.1, 0.95, 400)
        Wq_theory = rho_dense / (mu * (1 - rho_dense))
        Lq_theory = (rho_dense ** 2) / (1 - rho_dense)

        plot_lines(rho, wq_m,
                   "M/M/1 Waiting Time: Simulation vs Theory",
                   "Utilization (rho)", "Average Waiting Time (Wq)",
                   f"{out_prefix}_wait_theory.png",
                   theory_x=rho_dense, theory_y=Wq_theory,
                   yerr=wq_ci, legend_labels=["Simulation (±95% CI)", "Theory Wq"])

        plot_lines(rho, lq_m,
                   "M/M/1 Queue Length: Simulation vs Theory",
                   "Utilization (rho)", "Average Queue Length (Lq)",
                   f"{out_prefix}_queue_theory.png",
                   theory_x=rho_dense, theory_y=Lq_theory,
                   yerr=lq_ci, legend_labels=["Simulation (±95% CI)", "Theory Lq"])


def main():
    ap = argparse.ArgumentParser(description="M/M/1 simulator (discrete-event, no external sim libs).")
    ap.add_argument("--mode", choices=["single", "multi"], default="multi",
                    help="single = one run per lambda; multi = multi-seed with 95%% CIs + theory overlays")
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--sim-time", type=float, default=30000.0)
    ap.add_argument("--warmup", type=float, default=3000.0)
    ap.add_argument("--points", type=int, default=12, help="number of lambda points in sweep")
    ap.add_argument("--seeds", type=int, default=20, help="only used in multi mode")
    ap.add_argument("--seed0", type=int, default=2025, help="base seed")
    ap.add_argument("--out", type=str, default="mm1_output", help="output prefix (files will get this prefix)")
    args = ap.parse_args()

    if args.mode == "single":
        run_single(mu=args.mu, sim_time=args.sim_time, warmup=args.warmup,
                   n_points=args.points, seed=args.seed0, out_prefix=args.out+"_single")
    else:
        run_multi(mu=args.mu, sim_time=args.sim_time, warmup=args.warmup,
                  n_points=args.points, seeds=args.seeds, seed0=args.seed0, out_prefix=args.out+"_multi")


if __name__ == "__main__":
    main()
