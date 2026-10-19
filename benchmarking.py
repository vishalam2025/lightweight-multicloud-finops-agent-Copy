import time
INSTANCE_CATALOG = {
    "t2.micro":  {"vcpu": 1, "hourly_usd": 0.0116},
    "t3.micro":  {"vcpu": 2, "hourly_usd": 0.0104},
    "t3.small":  {"vcpu": 2, "hourly_usd": 0.0208},
    "t3.medium": {"vcpu": 2, "hourly_usd": 0.0416},
    "t2.medium": {"vcpu": 2, "hourly_usd": 0.0464},
}
def cpu_task(iterations):
    count = 0
    n = 2
    while count < iterations:
        is_prime = True
        for d in range(2, int(n ** 0.5) + 1):
            if n % d == 0:
                is_prime = False
                break
        if is_prime:
            count += 1
        n += 1
    return n
def run_benchmark(base_iterations=3000):
    results = []
    max_vcpu = max(spec["vcpu"] for spec in INSTANCE_CATALOG.values())

    for instance_type, spec in INSTANCE_CATALOG.items():
        capacity_factor = max_vcpu / spec["vcpu"]

        start = time.perf_counter()
        cpu_task(base_iterations)
        raw_time = time.perf_counter() - start

        emulated_time = raw_time * capacity_factor
        cost_per_run = (emulated_time / 3600) * spec["hourly_usd"]
        efficiency_score = emulated_time * spec["hourly_usd"]

        results.append({
            "instance_type": instance_type,
            "vcpu": spec["vcpu"],
            "hourly_usd": spec["hourly_usd"],
            "execution_time_sec": round(emulated_time, 4),
            "cost_per_run_usd": round(cost_per_run, 8),
            "efficiency_score": round(efficiency_score, 6),
        })

    results.sort(key=lambda r: r["efficiency_score"])
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    return results
if __name__ == "__main__":
    for row in run_benchmark():
        print(row)
