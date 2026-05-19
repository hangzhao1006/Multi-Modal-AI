# Paste this entire file as a single cell at the bottom of build_phase_arbitrator.ipynb.
# Reuses variables defined earlier in that notebook (imu, raw, mag, mag_dd,
# energy_rate, pa, demo_actions, load_imu_sample, IMU_LEN, T_I).

import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

BUNDLE_DIR = "/content/phase_archive"
os.makedirs(BUNDLE_DIR, exist_ok=True)

fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
t = np.arange(IMU_LEN)
axes[0].plot(t, mag[0, 0].numpy())
axes[0].set_title("|a(t)| acceleration magnitude (throw)")
axes[1].plot(t, mag_dd[0, 0].numpy(), color="C1")
axes[1].set_title("d2|a|/dt2 2nd derivative")
axes[2].plot(t, energy_rate[0, 0].numpy(), color="C2")
axes[2].set_title("d/dt(|a|^2) energy rate")
axes[2].set_xlabel("timestep")
for ax in axes:
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BUNDLE_DIR, "phase_features_throw.png"), dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(10, 4))
action_stats = []
for action_id, label in demo_actions:
    try:
        imu_sample = load_imu_sample(action_id, subject=1, trial=1)
        a = pa(imu_sample).squeeze().detach().numpy()
        ax.plot(np.linspace(0, 1, T_I), a, "o-", label=label)
        action_stats.append((action_id, label, float(a.mean()), float(a.min()), float(a.max())))
    except Exception as e:
        action_stats.append((action_id, label, None, None, str(e)))
ax.axhline(0.5, ls="--", color="gray", alpha=0.5)
ax.set_ylim(0, 1)
ax.set_xlabel("normalized time")
ax.set_ylabel("alpha (1=vision, 0=IMU)")
ax.set_title("alpha(t) untrained (random weights)")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower center")
plt.tight_layout()
plt.savefig(os.path.join(BUNDLE_DIR, "alpha_untrained.png"), dpi=120)
plt.close()

lines = []
lines.append("# Phase Arbitrator - Untrained Baseline\n\n")
lines.append("Module params: " + f"{sum(p.numel() for p in pa.parameters()):,}" + "\n")
lines.append("Output shape:  (B, " + str(T_I) + ")\n")
lines.append("Output range:  alpha in [0, 1] (Sigmoid)\n\n")
lines.append("## Throw sample physical features (sanity check)\n")
lines.append(f"  |a(t)|       min={mag[0,0].numpy().min():.3f}  max={mag[0,0].numpy().max():.3f}  mean={mag[0,0].numpy().mean():.3f}\n")
lines.append(f"  d2|a|/dt2    min={mag_dd[0,0].numpy().min():.3f}  max={mag_dd[0,0].numpy().max():.3f}\n")
lines.append(f"  energy_rate  min={energy_rate[0,0].numpy().min():.3f}  max={energy_rate[0,0].numpy().max():.3f}\n\n")
lines.append("## Untrained alpha(t) stats per demo action (expect ~0.5)\n")
for action_id, label, mean, mn, mx in action_stats:
    if mean is None:
        lines.append(f"  a{action_id}  {label}  ERROR: {mx}\n")
    else:
        lines.append(f"  a{action_id}  {label}\n")
        lines.append(f"           mean={mean:.3f}  min={mn:.3f}  max={mx:.3f}\n")

with open(os.path.join(BUNDLE_DIR, "phase_results.txt"), "w") as f:
    f.writelines(lines)

print("Bundle contents:")
for fn in sorted(os.listdir(BUNDLE_DIR)):
    sz = os.path.getsize(os.path.join(BUNDLE_DIR, fn)) / 1024
    print(f"  {fn}  ({sz:.1f} KB)")

zip_path = "/content/phase_archive.zip"
shutil.make_archive(zip_path.replace(".zip", ""), "zip", BUNDLE_DIR)
print("\nzipped:", zip_path, f"({os.path.getsize(zip_path)/1024:.1f} KB)")

from google.colab import files
files.download(zip_path)
