import matplotlib
matplotlib.use("Agg")
import json
import numpy as np
import matplotlib.pyplot as plt
import os

its_vals = [1, 10, 100, 1000]
backends = {
    'GPU-wg256': ('foreach_gpu_its{}_new', '-',  'o', 'tab:green'),
    'TBB':       ('foreach_tbb_its{}_new', '--', 's', 'tab:orange'),
    'GNU':       ('foreach_gnu_its{}_new', ':',  '^', 'tab:blue'),
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, its in enumerate(its_vals):
    ax = axes[idx]
    for label, (fname_tmpl, ls, marker, color) in backends.items():
        fname = fname_tmpl.format(its)
        path = f'/home/hemali/results/{fname}.json'
        d = json.load(open(path))
        sizes, medians, q25s, q75s = [], [], [], []
        times = {}
        for b in d['benchmarks']:
            if b.get('run_type') == 'aggregate':
                continue
            n = int(b['name'].split('/')[-2])
            times.setdefault(n, []).append(b['real_time'] / 1e6)
        for n in sorted(times.keys()):
            t = times[n][1:] if len(times[n]) >= 3 else times[n]
            sizes.append(n)
            medians.append(np.median(t))
            q25s.append(np.percentile(t, 25))
            q75s.append(np.percentile(t, 75))
        line, = ax.plot(sizes, medians, marker=marker, markersize=4,
                        label=label, linestyle=ls, linewidth=1.5, color=color)
        ax.fill_between(sizes, q25s, q75s, alpha=0.15, color=color)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_title(f'kernel_its={its}', fontsize=11)
    ax.set_xlabel('Input size (elements)')
    ax.set_ylabel('Time (ms)')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

fig.suptitle('X::for_each - kernel_its comparison (GPU-wg256 vs TBB vs GNU)', fontsize=13)
plt.tight_layout()
out = '/home/hemali/pstl-bench/plots_stats/foreach_its_comparison.png'
plt.savefig(out, dpi=150)
plt.close()
print(f'Saved {out}')
