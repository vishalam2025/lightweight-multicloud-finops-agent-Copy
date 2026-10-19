import json
import math
import time
from concurrent.futures import ProcessPoolExecutor


def count_primes(limit):
    count = 0

    for number in range(2, limit):
        is_prime = True

        for divisor in range(2, math.isqrt(number) + 1):
            if number % divisor == 0:
                is_prime = False
                break

        if is_prime:
            count += 1

    return count


if __name__ == "__main__":
    start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=2) as pool:
        counts = list(pool.map(count_primes, [400_000, 400_000]))

    elapsed = time.perf_counter() - start

    print(json.dumps({
        "execution_time_sec": elapsed,
        "workers": 2,
        "limit_per_worker": 400_000,
        "prime_counts": counts,
    }))
