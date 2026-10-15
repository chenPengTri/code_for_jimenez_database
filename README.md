# Channel-flow velocity reconstruction and CPU/GPU comparison

These three scripts read the supplied Fortran channel-flow snapshots, reconstruct
velocity, and compare CPU and GPU results for the same snapshot.

| Script | Purpose | Dependencies |
| --- | --- | --- |
| `field_to_velocity.py` | CPU reconstruction using a sparse Chebyshev tau solver | NumPy, SciPy |
| `field_to_velocity_gpu.py` | Batched reconstruction, Chebyshev evaluation and FFTs on an NVIDIA CUDA GPU | NumPy, CuPy; NumPy alone for CPU verification mode |
| `compare_velocity.py` | Compare velocity arrays, mean profiles and spectra | NumPy, Matplotlib |

Place the three scripts in your working directory. The GPU and comparison scripts
are standalone: neither imports `field_to_velocity.py`.

## 1. Installation

A Python 3.10+ environment is suitable for current CuPy releases. The scripts
use Python 3.7-compatible syntax, but older Python environments require compatible
older dependency versions.

CPU reconstruction and comparison:

```bash
python3 -m pip install numpy scipy matplotlib
```

For an NVIDIA GPU with a driver compatible with CUDA 12:

```bash
python3 -m pip install "cupy-cuda12x[ctk]"
```

Select the CuPy package appropriate for your CUDA/driver setup. Install only one
CuPy distribution in an environment. A supported NVIDIA GPU and driver are
required for this script's GPU mode.

Dependency and driver guidance:
[CuPy installation documentation](https://docs.cupy.dev/en/stable/install.html).

## 2. Input format and reconstruction conventions

The input is a Fortran sequential unformatted file with 4-byte record markers:

1. Parameters: `time, Re, alp, bet, a0, mx, my, mz`.
2. Interleaved mean coefficients: `u00(1), w00(1), u00(2), w00(2), ...`.
3. One `vor, phi` plane record per Chebyshev index `j`.

The writer therefore produces **`my + 2` records**, despite the description
calling the last group “record #3.” All real values are 4-byte except `time`,
which may be 4- or 8-byte; both converters detect its size from the header length.
Integers are 4-byte. Big-endian is the default. Eight-byte record markers and
split Fortran subrecords are not supported.

Adjacent real entries represent real and imaginary parts of a Fourier
coefficient. The scripts assume:

- `mx` is even and `mz` is odd.
- Streamwise modes are `0, ..., mx/2 - 1`.
- Spanwise modes are `0, 1, ..., K, -K, ..., -1`, with `K=(mz-1)/2`.
- Fourier coefficients are normalized series coefficients for
  `exp(+i*(kx*x + kz*z))`, with `kx=p*alp` and `kz=k*bet`.
- Chebyshev coefficients directly multiply `T_n(y)` without endpoint half weights.
- `vor` is wall-normal vorticity, `omega_y = du/dz - dw/dx`.
- `phi = laplacian(v)`; the walls are at centered `y/h = -1` and `+1`.

For nonzero `q=kx**2+kz**2`, reconstruction uses

```text
(D_y**2 - q) v_hat = phi_hat,    v_hat(-1) = v_hat(+1) = 0
u_hat = i * (kx*D_y(v_hat) - kz*omega_y_hat) / q
w_hat = i * (kz*D_y(v_hat) + kx*omega_y_hat) / q
```

At the zero horizontal mode, `u00` and `w00` supply the mean velocities and
`v_hat=0`. The supplied fields must be consistent with the remaining channel
boundary conditions; the reconstruction does not impose an additional independent
condition on `D_y(v_hat)`.

The two converters retain the same transform conventions. CPU/GPU agreement
alone cannot establish that those conventions match an undocumented database
normalization. Validate them against the source data or a known physical profile.

## 3. Convert one snapshot on the CPU

```bash
python3 field_to_velocity.py \
    Re550_2pipi/Re5502pipi.0050 \
    --out ./TS0050_CPU
```

The script prints header parameters, reading progress, reconstruction progress,
output size and elapsed time. Progress updates occur at stage boundaries and
approximately every five seconds while a loop is advancing.

## 4. Convert the same snapshot on the GPU

First test the selected GPU backend:

```bash
python3 field_to_velocity_gpu.py --self-test
```

This compares batched Helmholtz solutions, Chebyshev evaluation and inverse FFTs
against independent NumPy reference calculations. Then run:

```bash
python3 field_to_velocity_gpu.py \
    Re550_2pipi/Re5502pipi.0050 \
    --out ./TS0050_GPU
```

The GPU script uses double precision for the Helmholtz solves and defaults to
single precision for Chebyshev evaluation and FFTs. Add `--fft-double` for double
precision transforms. Intermediate reconstructed coefficients and output velocity
files remain single precision in either case.

The solver batches Fourier modes; transforms process one velocity component at
a time and batches of wall-normal planes. Reading and output-file writing remain
on the CPU. Speed depends on GPU performance, transfers and disk throughput.

To reduce temporary GPU memory use:

```bash
python3 field_to_velocity_gpu.py \
    Re550_2pipi/Re5502pipi.0050 \
    --out ./TS0050_GPU \
    --mode-batch 2048 --y-batch 1
```

Reducing batches does not remove the need to hold one complete component's
spectral coefficients on the GPU during transforms. Host memory also holds the
input coefficients and reconstructed coefficients.

For CPU verification of the batched algorithm without CuPy:

```bash
python3 field_to_velocity_gpu.py --backend numpy --self-test
python3 field_to_velocity_gpu.py \
    Re550_2pipi/Re5502pipi.0050 \
    --backend numpy --out ./TS0050_BATCHED_CPU
```

## 5. Converter options

Both converters accept:

| Option | Default | Meaning |
| --- | --- | --- |
| `field` | Required | Input Fortran filename |
| `--out` | `velocity` | Output directory |
| `--endian` | `>` | Big-endian `>` or little-endian `<` |
| `--nx` | `ceil(1.5*mx)` | Streamwise output points; must be at least `mx` |
| `--nz` | `ceil(1.5*mz)` | Spanwise output points; must be at least `mz` |
| `--lab-frame` | Off | Add `a0` to `u`, assuming `U_lab = U_stored + a0` |
| `--half-cheb-endpoints` | Off | Divide the first and last Chebyshev coefficients by two before reconstruction |

Quote endianness on the shell so `>` is not interpreted as redirection:

```bash
python3 field_to_velocity.py input_field --endian '>' --out velocity
```

Use `--half-cheb-endpoints` only when the source convention calls for those
weights. It is not a general correction for an unexpected profile.

GPU-only options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--backend` | `cupy` | GPU execution or `numpy` CPU verification |
| `--device` | `0` | CUDA device index |
| `--mode-batch` | `8192` | Fourier modes per reconstruction batch |
| `--y-batch` | `4` | Wall-normal planes per transform batch |
| `--fft-double` | Off | Double precision Chebyshev evaluation and FFTs |
| `--self-test` | Off | Run numerical checks without an input file |

## 6. Output arrays and coordinates

Each converter produces:

| File | Shape / contents |
| --- | --- |
| `u.npy`, `v.npy`, `w.npy` | `float32`, shape `(nx, my, nz)`, axes `(x,y,z)` |
| `x.npy` | Streamwise coordinate, endpoint excluded |
| `y.npy` | Chebyshev nodes, descending from `+1` to `-1` |
| `z.npy` | Spanwise coordinate, endpoint excluded |
| `metadata.json` | Parameters, array shape and conversion settings |

Lengths are in channel-half-height units: `Lx/h=2*pi/alp`,
`Lz/h=2*pi/bet`. Velocities retain the source-file scaling; they are not
automatically converted to wall units or bulk-velocity units.

Load a field without copying the complete array into RAM:

```python
import numpy as np

folder = "TS0050_GPU"
u = np.load(folder + "/u.npy", mmap_mode="r")
v = np.load(folder + "/v.npy", mmap_mode="r")
w = np.load(folder + "/w.npy", mmap_mode="r")
y = np.load(folder + "/y.npy")

iy = np.argmin(np.abs(y - 0.15))
u_plane = np.asarray(u[:, iy, :], dtype=np.float64)
u_prime = u_plane - u_plane.mean()
```

Reusing a converter output directory overwrites its output arrays. Use separate
CPU and GPU directories when comparing results.

The three velocity arrays require `3*nx*my*nz*4` bytes on disk. For
`(1536,257,1535)`, this is approximately **6.772 GiB per snapshot**. The 46
snapshots in `20:920:20` require approximately **312 GiB** for one conversion
set, excluding input data and comparison products. Caching all 46 `u` arrays in
RAM would require approximately **104 GiB**.

## 7. Compare CPU and GPU results

Use outputs from the same snapshot, grid, reference frame and coefficient
conventions:

```bash
python3 compare_velocity.py \
    ./TS0050_CPU ./TS0050_GPU \
    --y 0.15 --out ./comparison_TS0050
```

The script checks available metadata and coordinate agreement, then compares
every value in `u`, `v` and `w` in chunks. It reports maximum absolute error,
RMS error, relative L2 error, failed-value counts and mean-profile differences.
Accumulation uses double precision.

Default pointwise consistency criterion, using CPU values as reference:

```text
abs(GPU - CPU) <= 1e-6 + 1e-4*abs(CPU)
```

The mean profiles are horizontal spatial averages at each `y`, not time or
ensemble averages. At the selected plane, each component's own horizontal mean
is removed before computing spectra.

### Location of y/h = 0.15

By default, the location is **centered `y/h=0.15`**, using the coordinates saved
by the converters. The nearest existing Chebyshev node is selected; no
interpolation is performed. The actual coordinate and index are printed and
stored in the report.

For `0.15h` measured from the lower wall, corresponding to centered `y/h=-0.85`:

```bash
python3 compare_velocity.py TS0050_CPU TS0050_GPU \
    --y 0.15 --y-origin lower-wall --out comparison_lower_wall
```

Use `--y-origin upper-wall` for a distance measured from the upper wall.

### Spectral definitions and checks

The two-dimensional periodogram uses angular wavenumbers:

```text
F = fft2(component - plane_mean) / (nx*nz)
S(kx,kz) = abs(F)**2 / (dkx*dkz)
dkx = 2*pi/Lx;  dkz = 2*pi/Lz
sum(S)*dkx*dkz = plane fluctuation variance
```

There is no factor of `1/2`, window, spectral smoothing or snapshot averaging.
The periodic domain permits an unwindowed transform. Signed two-dimensional
spectra retain both positive and negative wavenumbers. One-dimensional spectra
integrate over the other direction and fold negative wavenumbers onto positive
ones, counting the zero and even-grid Nyquist modes only once.

The script verifies the variance identity (Parseval) independently for each
component and output. Spectral agreement requires:

```text
sum(abs(S_gpu-S_cpu))*dkx*dkz <= 1e-12 + 1e-4*variance_cpu
```

The 2D plots use common CPU/GPU color scales per component. Difference maps show
`(S_gpu-S_cpu)/peak(S_cpu,S_gpu)`. Large maps are block-averaged for display;
saved spectral arrays retain full resolution. These are single-snapshot spectra,
so they need not be smooth turbulent ensemble spectra.

### Comparison outputs

| File | Contents |
| --- | --- |
| `report.json` | Per-component errors, selected plane, tolerances and overall consistency |
| `mean_profiles.csv` | Centered `y/h`, CPU/GPU means of `u`, `v`, `w` |
| `mean_profiles.png` | Mean profiles and CPU/GPU differences |
| `spectra_1d.png` | One-sided streamwise and spanwise spectra for all components |
| `spectra_2d.png` | Signed x-z density maps and differences |
| `spectra_u.npz`, `spectra_v.npz`, `spectra_w.npz` | Full-resolution densities, wavenumbers, one-dimensional spectra and plane location |

The script exits with code `0` when all field, spectral and Parseval checks pass,
and `1` for a reported consistency failure. Input errors raise exceptions.
Mean-profile errors are reported; there is no separate mean-profile pass limit.

Comparison options:

| Option | Default | Meaning |
| --- | --- | --- |
| `cpu`, `gpu` | Required | Converter output directories |
| `--out` | `comparison` | Report/plot directory |
| `--y` | `0.15` | Requested normalized location |
| `--y-origin` | `center` | `center`, `lower-wall`, or `upper-wall` |
| `--rtol`, `--atol` | `1e-4`, `1e-6` | Pointwise relative and absolute tolerances |
| `--psd-rtol` | `1e-4` | Relative integrated spectral-error tolerance |
| `--variance-atol` | `1e-12` | Absolute integrated spectral-error tolerance |
| `--chunk-x` | `8` | Streamwise slabs read per comparison chunk |

## 8. Convert snapshots 0020 through 0920

Save this as `run_fields.sh` and run `bash run_fields.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

for ((n=20; n<=920; n+=20)); do
    printf -v tag "%04d" "$n"
    echo "Processing ${tag} -> TS${tag}"
    python3 field_to_velocity_gpu.py \
        "Re550_2pipi/Re5502pipi.${tag}" \
        --out "./TS${tag}"
done
```

Replace `field_to_velocity_gpu.py` with `field_to_velocity.py` for CPU conversion.
The loop includes both endpoints, runs sequentially, and stops on a conversion
failure. Redirect output to preserve a log if desired:

```bash
bash run_fields.sh > conversion.log 2>&1
```

## 9. Troubleshooting and validation status

- **Record-size mismatch:** confirm endianness, numerical sizes and record-marker
  sizes. Use `--endian '<'` only for little-endian data.
- **CUDA initialization or missing CuPy:** activate the intended environment and
  check the driver and CuPy installation. Run the GPU script's `--self-test`.
- **GPU out of memory:** reduce `--mode-batch` and `--y-batch`; omit `--fft-double`
  when single precision transforms are sufficient. One full component's spectral
  array must still fit on the device during transforms.
- **Unexpected velocity scaling:** verify the Fourier/Chebyshev conventions and
  whether `--lab-frame` is appropriate. `Re` is not automatically a bulk Reynolds
  number, and no velocity-unit conversion is applied.
- **CPU/GPU comparison fails:** first confirm matching snapshots and settings;
  inspect full-field and spectral errors. `--fft-double` can help assess transform
  roundoff. Choose tolerances based on numerical accuracy requirements.
- **Older NumPy:** the converters explicitly multiply inverse FFTs by transform
  lengths instead of using the newer `norm="forward"` option. The `yield` syntax
  is also compatible with Python 3.7.

The CPU converter was tested against a known synthetic velocity field. The
batched solver was compared against the CPU tau solver, including 257
Chebyshev coefficients and large wavenumbers. Comparison tests covered synthetic
outputs, analytic spectral peaks, Parseval normalization, Nyquist folding and
injected-error detection. CUDA hardware was unavailable during development:
actual GPU execution, performance and database snapshot consistency must be
checked on the target machine. CPU/GPU agreement is a numerical consistency
check, not independent validation of the source database conventions.

Display command-line help for any script with `--help`.
