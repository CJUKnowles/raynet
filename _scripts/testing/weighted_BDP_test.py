import numpy as np
import matplotlib.pyplot as plt

def weighted_network_params(bw_range, rtt_range):
    samples = []

    bdp_min = bw_range[0] * rtt_range[0]
    bdp_max = bw_range[1] * rtt_range[1]

    def sample_linear_bdp(low, high):
        u = np.random.rand()
        return low + (high - low) * np.sqrt(u)  # bias toward large

    working = True
    while working:
        # Step 1: sample BDP globally with desired bias
        bdp = sample_linear_bdp(bdp_min, bdp_max)

        # Step 2: derive feasible bw range for this BDP
        bw_low = max(bw_range[0], bdp / rtt_range[1])
        bw_high = min(bw_range[1], bdp / rtt_range[0])

        # skip impossible cases
        if bw_low > bw_high:
            continue
        working = False
        # Step 3: sample bw conditionally
        bw = np.random.uniform(bw_low, bw_high)

        # Step 4: derive RTT
        rtt = bdp / bw
        bdp = rtt * bw
    return bw, rtt, bdp


# Tester
if __name__ == "__main__":
    bw_range = (5, 100)
    rtt_range = (40, 70)

    samples = []
    for i in range(10000):
        samples.append(weighted_network_params(bw_range, rtt_range))

    samples = np.array(samples)

    bw_vals = samples[:,0]
    rtt_vals = samples[:,1]
    bdp_vals = samples[:,2]

    plt.figure(figsize=(10,6))

    scatter = plt.scatter(
        rtt_vals,
        bw_vals,
        c=bdp_vals,
        s=5,
        alpha=0.5
    )

    plt.xlabel("RTT (ms)")
    plt.ylabel("Bandwidth (Mbps)")
    plt.title("High-BDP Weighted Sampling")

    cbar = plt.colorbar(scatter)
    cbar.set_label("BDP")

    plt.grid(True)
    plt.savefig("./_scripts/testing/weighted_BDP_sampling.png")
    plt.close()



    plt.figure(figsize=(10, 5))

    plt.hist(
        bw_vals,
        bins=16
    )

    plt.xlabel("Bandwidth (Mbps)")
    plt.ylabel("Count")
    plt.title("Histogram of Sampled BWs")

    plt.grid(True)
    plt.savefig("./_scripts/testing/weighted_bw_histogram.png")
    plt.close()

    plt.figure(figsize=(10, 5))

    plt.hist(
        rtt_vals,
        bins=16
    )

    plt.xlabel("RTT")
    plt.ylabel("Count")
    plt.title("Histogram of Sampled RTTs")

    plt.grid(True)
    plt.savefig("./_scripts/testing/weighted_rtt_histogram.png")
    plt.close()

    plt.figure(figsize=(10, 5))

    plt.hist(
        bdp_vals,
        bins=16
    )

    plt.xlabel("BDP")
    plt.ylabel("Count")
    plt.title("Histogram of Sampled BDPs")

    plt.grid(True)
    plt.savefig("./_scripts/testing/weighted_BDP_histogram.png")
    plt.close()