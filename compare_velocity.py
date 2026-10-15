"""Compare field_to_velocity.py and field_to_velocity_gpu.py output folders.

Requires NumPy and Matplotlib. Compatible with Python 3.7 syntax.
Coordinates saved by the converters use centered y/h in [-1,1].
Spectra are single-snapshot, plane-mean-subtracted periodograms, without a
window (the horizontal directions are periodic). All spectral densities use
angular wavenumbers, not cycles per unit length; no factor of 1/2 is applied.
"""
import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def ratio(a, b):
    return float(a / b) if b > 0 else (0.0 if a == 0 else float("inf"))


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def validate_coordinates(cpu_dir, gpu_dir):
    coords = []
    for name in ("x", "y", "z"):
        a = np.load(cpu_dir / (name + ".npy"))
        b = np.load(gpu_dir / (name + ".npy"))
        if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all():
            raise ValueError("Invalid coordinate array: " + name)
        if a.shape != b.shape or not np.allclose(a, b, rtol=1e-10, atol=1e-12):
            raise ValueError("CPU/GPU coordinate mismatch: " + name)
        if name != "y":
            d = np.diff(a)
            if d[0] <= 0 or not np.allclose(d, d[0], rtol=1e-7, atol=1e-12):
                raise ValueError(name + " coordinates must be increasing and uniformly spaced")
        coords.append(a)
    return coords


def check_metadata(cpu_dir, gpu_dir):
    """Catch different snapshots, reference frames, or coefficient conventions."""
    paths = [d / "metadata.json" for d in (cpu_dir, gpu_dir)]
    if not all(p.exists() for p in paths):
        print("Note: metadata missing; snapshot/reference-frame identity cannot be checked.", flush=True)
        return
    a, b = [json.loads(p.read_text()) for p in paths]
    for key in ("time", "Re", "alp", "bet", "a0", "mx", "my", "mz",
                "lab_frame", "half_cheb_endpoints"):
        if key in a and key in b and a[key] != b[key]:
            raise ValueError("Metadata mismatch for {}: {} vs {}".format(key, a[key], b[key]))


def compare_component(a, b, rtol, atol, chunk_x):
    """Full-domain metrics and horizontal means, using float64 accumulation."""
    nx, ny, nz = a.shape
    sa, sb = np.zeros(ny), np.zeros(ny)
    squared_error = squared_a = squared_b = dot = 0.0
    max_error = max_abs_a = max_abs_b = 0.0
    failures = 0
    last = monotonic()
    for start in range(0, nx, chunk_x):
        stop = min(start + chunk_x, nx)
        aa = np.asarray(a[start:stop], dtype=np.float64)
        bb = np.asarray(b[start:stop], dtype=np.float64)
        if not np.isfinite(aa).all() or not np.isfinite(bb).all():
            raise ValueError("Nonfinite values in velocity fields")
        diff = bb - aa
        absolute = np.abs(diff)
        max_error = max(max_error, float(absolute.max()))
        max_abs_a = max(max_abs_a, float(np.abs(aa).max()))
        max_abs_b = max(max_abs_b, float(np.abs(bb).max()))
        failures += int(np.count_nonzero(absolute > atol + rtol * np.abs(aa)))
        squared_error += float(np.sum(diff * diff))
        squared_a += float(np.sum(aa * aa))
        squared_b += float(np.sum(bb * bb))
        dot += float(np.sum(aa * bb))
        sa += aa.sum(axis=(0, 2))
        sb += bb.sum(axis=(0, 2))
        now = monotonic()
        if now - last >= 5 or stop == nx:
            print("  Full-field check: {}/{} x slabs ({:.1f}%)".format(stop, nx, 100 * stop / nx), flush=True)
            last = now
    mean_a, mean_b = sa / (nx * nz), sb / (nx * nz)
    metrics = dict(
        allclose=failures == 0, failed_values=failures, total_values=int(a.size),
        failed_fraction=failures / a.size,
        max_absolute_error=max_error, rms_error=float(np.sqrt(squared_error / a.size)),
        relative_l2_error=ratio(np.sqrt(squared_error), np.sqrt(squared_a)),
        max_error_over_cpu_peak=ratio(max_error, max_abs_a),
        rms_cpu=float(np.sqrt(squared_a / a.size)), rms_gpu=float(np.sqrt(squared_b / a.size)),
        cosine_similarity=ratio(dot, np.sqrt(squared_a * squared_b)),
        max_mean_profile_error=float(np.max(np.abs(mean_b - mean_a))),
    )
    return metrics, mean_a, mean_b


def spectrum2d(plane, dx, dz):
    """Two-sided density S with sum(S)*dkx*dkz == mean(plane_prime**2)."""
    plane = np.asarray(plane, dtype=np.float64)
    fluct = plane - plane.mean()
    nx, nz = plane.shape
    dkx, dkz = 2 * np.pi / (nx * dx), 2 * np.pi / (nz * dz)
    coefficients = np.fft.fft2(fluct) / (nx * nz)
    density = np.abs(coefficients)**2 / (dkx * dkz)
    return density, float(np.mean(fluct**2)), dkx, dkz


def fold_spectrum(values):
    """Fold negative wavenumbers, keeping DC and even-length Nyquist once."""
    n = len(values)
    indices = np.abs(np.rint(np.fft.fftfreq(n) * n).astype(int))
    out = np.zeros(n // 2 + 1)
    np.add.at(out, indices, values)
    return out


def plot_mean_profiles(out, y, profiles):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for col, name in enumerate(("u", "v", "w")):
        a, b = profiles[name]
        axes[0, col].plot(a, y, label="CPU")
        axes[0, col].plot(b, y, "--", label="GPU")
        axes[0, col].set_xlabel(r"$\langle " + name + r"\rangle_{xz}$")
        axes[0, col].legend()
        axes[1, col].plot(b - a, y, color="tab:red")
        axes[1, col].set_xlabel("GPU - CPU mean " + name)
        for row in range(2):
            axes[row, col].grid(alpha=.3)
    axes[0, 0].set_ylabel("Centered y/h")
    axes[1, 0].set_ylabel("Centered y/h")
    fig.suptitle("Horizontal mean velocity profiles (single snapshot)")
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(out / "mean_profiles.png", dpi=180)
    plt.close(fig)


def plot_spectra(out, spectra, kx, kz, dkx, dkz, actual_y):
    kxp = np.arange(len(kx) // 2 + 1) * dkx
    kzp = np.arange(len(kz) // 2 + 1) * dkz
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    maps, axs = plt.subplots(3, 3, figsize=(15, 12))
    for row, name in enumerate(("u", "v", "w")):
        sa, sb = spectra[name]
        exa, exb = fold_spectrum(sa.sum(axis=1) * dkz), fold_spectrum(sb.sum(axis=1) * dkz)
        eza, ezb = fold_spectrum(sa.sum(axis=0) * dkx), fold_spectrum(sb.sum(axis=0) * dkx)
        for col, k, ea, eb, direction in ((0, kxp, exa, exb, "x"), (1, kzp, eza, ezb, "z")):
            ax = axes[row, col]
            peak = max(float(ea.max()), float(eb.max()))
            for e, style, label in ((ea, "-", "CPU"), (eb, "--", "GPU")):
                mask = (k > 0) & (e > max(peak * 1e-14, np.finfo(float).tiny))
                ax.loglog(k[mask], e[mask], style, label=label)
            ax.set_xlabel(r"$k_" + direction + r"h$")
            ax.set_ylabel(r"$E_{" + name + name + r"}^{(" + direction + r")}$")
            ax.grid(True, which="both", alpha=.2)
            ax.legend()
        np.savez_compressed(out / ("spectra_" + name + ".npz"),
                            kx=kx, kz=kz, dkx=dkx, dkz=dkz,
                            psd_cpu=sa, psd_gpu=sb, kx_one_sided=kxp, kz_one_sided=kzp,
                            Ex_cpu=exa, Ex_gpu=exb, Ez_cpu=eza, Ez_gpu=ezb,
                            y_over_h=actual_y)
        # Block-average only the PNG maps; full-resolution densities are saved.
        # Averaging retains narrow peaks that stride-based sampling could miss.
        sx, sz = max(1, int(np.ceil(len(kx) / 400))), max(1, int(np.ceil(len(kz) / 400)))
        ix, iz = np.arange(0, len(kx), sx), np.arange(0, len(kz), sz)
        cx, cz = np.minimum(sx, len(kx) - ix), np.minimum(sz, len(kz) - iz)
        xs = np.add.reduceat(np.fft.fftshift(kx), ix) / cx
        zs = np.add.reduceat(np.fft.fftshift(kz), iz) / cz
        shifted = []
        for s in (sa, sb):
            block = np.add.reduceat(np.add.reduceat(np.fft.fftshift(s), ix, axis=0), iz, axis=1)
            shifted.append((block / (cx[:, None] * cz[None, :])).T)
        peak = max(float(sa.max()), float(sb.max()), np.finfo(float).tiny)
        norm = LogNorm(vmin=max(peak * 1e-10, np.finfo(float).tiny), vmax=peak * (1 + 1e-12))
        for col, values in enumerate(shifted):
            im = axs[row, col].pcolormesh(xs, zs, np.maximum(values, norm.vmin),
                                         shading="auto", norm=norm, cmap="viridis", rasterized=True)
            axs[row, col].set_title(name + (": CPU PSD" if col == 0 else ": GPU PSD"))
            maps.colorbar(im, ax=axs[row, col])
        delta = (shifted[1] - shifted[0]) / peak
        limit = max(float(np.max(np.abs(delta))), 1e-15)
        im = axs[row, 2].pcolormesh(xs, zs, delta, shading="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        axs[row, 2].set_title(name + ": (GPU - CPU) / peak PSD")
        maps.colorbar(im, ax=axs[row, 2])
        for col in range(3):
            axs[row, col].set_xlabel(r"$k_xh$")
            axs[row, col].set_ylabel(r"$k_zh$")
    fig.suptitle("One-sided fluctuation spectra at centered y/h = {:.6f}".format(actual_y))
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(out / "spectra_1d.png", dpi=180)
    plt.close(fig)
    maps.suptitle("Two-sided x-z fluctuation PSD at centered y/h = {:.6f}".format(actual_y))
    maps.tight_layout(rect=[0, 0, 1, .97])
    maps.savefig(out / "spectra_2d.png", dpi=160)
    plt.close(maps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu", type=Path, help="Output folder from field_to_velocity.py")
    parser.add_argument("gpu", type=Path, help="Output folder from field_to_velocity_gpu.py")
    parser.add_argument("--out", type=Path, default=Path("comparison"))
    parser.add_argument("--y", type=float, default=.15, help="Requested y/h (default .15)")
    parser.add_argument("--y-origin", choices=("center", "lower-wall", "upper-wall"), default="center")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Pointwise relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-6, help="Pointwise absolute velocity tolerance")
    parser.add_argument("--psd-rtol", type=float, default=1e-4, help="Relative integrated absolute PSD error tolerance")
    parser.add_argument("--variance-atol", type=float, default=1e-12, help="Absolute integrated PSD error tolerance")
    parser.add_argument("--chunk-x", type=int, default=8, help="x slabs loaded at once")
    args = parser.parse_args()
    if args.chunk_x < 1 or min(args.rtol, args.atol, args.psd_rtol, args.variance_atol) < 0:
        parser.error("chunk-x must be positive and tolerances nonnegative")
    started = monotonic()
    check_metadata(args.cpu, args.gpu)
    x, y, z = validate_coordinates(args.cpu, args.gpu)
    target = args.y if args.y_origin == "center" else (-1 + args.y if args.y_origin == "lower-wall" else 1 - args.y)
    if not np.isfinite(target) or target < min(y) or target > max(y):
        parser.error("Requested y lies outside the coordinate range")
    iy = int(np.argmin(np.abs(y - target)))
    print("Requested {} y/h={:.6f}; nearest centered y/h={:.6f}, iy={}".format(
        args.y_origin, args.y, y[iy], iy), flush=True)
    print("Pointwise criterion: |GPU-CPU| <= {} + {}*|CPU|".format(args.atol, args.rtol), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(cpu=str(args.cpu), gpu=str(args.gpu), requested_y=args.y, y_origin=args.y_origin,
                  selected_centered_y=float(y[iy]), iy=iy, rtol=args.rtol, atol=args.atol,
                  psd_rtol=args.psd_rtol, variance_atol=args.variance_atol, components={})
    profiles, spectra = {}, {}
    kx = 2 * np.pi * np.fft.fftfreq(len(x), d=x[1] - x[0])
    kz = 2 * np.pi * np.fft.fftfreq(len(z), d=z[1] - z[0])
    for name in ("u", "v", "w"):
        print("Comparing " + name, flush=True)
        a = np.load(args.cpu / (name + ".npy"), mmap_mode="r")
        b = np.load(args.gpu / (name + ".npy"), mmap_mode="r")
        if a.shape != (len(x), len(y), len(z)) or b.shape != a.shape:
            raise ValueError("Grid/velocity shape mismatch for " + name)
        stats, mean_a, mean_b = compare_component(a, b, args.rtol, args.atol, args.chunk_x)
        profiles[name] = (mean_a, mean_b)
        sa, va, dkx, dkz = spectrum2d(a[:, iy, :], x[1] - x[0], z[1] - z[0])
        sb, vb, _, _ = spectrum2d(b[:, iy, :], x[1] - x[0], z[1] - z[0])
        spectra[name] = (sa, sb)
        ia, ib = float(sa.sum() * dkx * dkz), float(sb.sum() * dkx * dkz)
        psd_error = float(np.sum(np.abs(sb - sa)) * dkx * dkz)
        psd_pass = psd_error <= args.variance_atol + args.psd_rtol * va
        parseval_a = abs(ia - va) <= 1e-12 + 1e-10 * va
        parseval_b = abs(ib - vb) <= 1e-12 + 1e-10 * vb
        stats.update(mean_cpu_at_y=float(mean_a[iy]), mean_gpu_at_y=float(mean_b[iy]),
                     variance_cpu=va, variance_gpu=vb, variance_relative_error=ratio(abs(vb - va), va),
                     integrated_psd_cpu=ia, integrated_psd_gpu=ib,
                     psd_integrated_absolute_error=psd_error, psd_relative_l1_error=ratio(psd_error, va),
                     psd_consistent=bool(psd_pass), parseval_cpu=bool(parseval_a), parseval_gpu=bool(parseval_b))
        report["components"][name] = stats
        print("  Field {} | max abs={:.3e}, RMS error={:.3e}, relative L2={:.3e}, failures={}".format(
            "PASS" if stats["allclose"] else "FAIL", stats["max_absolute_error"], stats["rms_error"],
            stats["relative_l2_error"], stats["failed_values"]), flush=True)
        print("  At plane: mean CPU/GPU={:.8g}/{:.8g}, variance={:.8g}/{:.8g}".format(
            mean_a[iy], mean_b[iy], va, vb), flush=True)
        print("  PSD {} | relative integrated error={:.3e}; Parseval CPU/GPU={}/{}".format(
            "PASS" if psd_pass else "FAIL", stats["psd_relative_l1_error"], parseval_a, parseval_b), flush=True)
        del a, b
    columns = [y]
    for name in ("u", "v", "w"):
        columns.extend(profiles[name])
    np.savetxt(args.out / "mean_profiles.csv", np.column_stack(columns), delimiter=",",
               header="y_centered_over_h,U_cpu,U_gpu,V_cpu,V_gpu,W_cpu,W_gpu", comments="")
    plot_mean_profiles(args.out, y, profiles)
    plot_spectra(args.out, spectra, kx, kz, dkx, dkz, float(y[iy]))
    report["consistent"] = all(s["allclose"] and s["psd_consistent"] and s["parseval_cpu"] and s["parseval_gpu"]
                               for s in report["components"].values())
    report["elapsed_seconds"] = monotonic() - started
    (args.out / "report.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    print("Overall: {}. Saved plots, spectra and report in {} ({:.1f}s)".format(
        "PASS" if report["consistent"] else "FAIL", args.out, report["elapsed_seconds"]), flush=True)
    return 0 if report["consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
