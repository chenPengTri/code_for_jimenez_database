"""GPU Fourier-Chebyshev channel velocity converter.
Default backend: CuPy/NVIDIA CUDA. --backend numpy is a CPU verification mode.
Conventions match field_to_velocity.py: normalized exp(+ikx*x+ikz*z) Fourier
series, direct T_n(y) coefficients, eta=du/dz-dw/dx, walls y=+/-1.
Requires Python >=3.7 and NumPy; GPU mode additionally requires compatible CuPy.
"""
import argparse
import json
import struct
from pathlib import Path
from time import monotonic
import numpy as np
from numpy.polynomial import chebyshev as ch

def progress(stage, total):
    """Print immediately, every five seconds, and at completion."""
    start = monotonic()
    last = start
    print(f"{stage}: 0/{total} (0%)", flush=True)

    def update(done):
        nonlocal last
        now = monotonic()
        if done == total or now - last >= 5.0:
            elapsed = now - start
            eta = elapsed * (total - done) / done if done else 0.0
            print(f"{stage}: {done}/{total} ({100.0 * done / total:.1f}%)"
                  f" | elapsed {elapsed:.1f}s | ETA {eta:.1f}s", flush=True)
            last = now
    return update

def read_field(filename, endian=">"):
    """Return metadata, mean Chebyshev coefficients, eta and phi.

    eta/phi shapes: (mx//2, mz, my), with complex Fourier coefficients.
    Supports 4-byte record markers and 4- or 8-byte time automatically.
    Split Fortran subrecords and 8-byte record markers are not supported.
    """
    if endian not in ("<", ">"):
        raise ValueError("endian must be '<' or '>'")
    print(f"Opening {filename} (endian={endian})", flush=True)
    with open(filename, "rb") as f:
        def record(expected=None):
            marker = f.read(4)
            if len(marker) != 4:
                raise EOFError("Missing Fortran record marker")
            size = struct.unpack(endian + "i", marker)[0]
            if size < 0:
                raise ValueError("Split Fortran subrecords are not supported")
            if expected is not None and size not in expected:
                raise ValueError(f"Unexpected record size {size}; expected {expected}. "
                                 "Check endianness and record-marker size.")
            payload = f.read(size)
            if len(payload) != size or f.read(4) != marker:
                raise ValueError("Truncated record or mismatched record markers")
            return payload

        header = record((32, 36))
        values = struct.unpack(endian + ("5f3i" if len(header) == 32 else "d4f3i"), header)
        names = ("time", "Re", "alp", "bet", "a0", "mx", "my", "mz")
        meta = dict(zip(names, values))
        mx, my, mz = (meta[n] for n in ("mx", "my", "mz"))
        print(f"Header: time={meta['time']:g}, Re={meta['Re']:g}, "
              f"mx={mx}, my={my}, mz={mz}; time uses "
              f"{8 if len(header) == 36 else 4} bytes", flush=True)
        if mx < 2 or mx % 2 or my < 4 or mz < 1 or mz % 2 != 1:
            raise ValueError("Require even mx >= 2, my >= 4 and odd mz")
        if meta["alp"] <= 0 or meta["bet"] <= 0:
            raise ValueError("alp and bet must be positive")
        dtype = np.dtype(endian + "f4")
        mean = np.frombuffer(record((8 * my,)), dtype).reshape(my, 2).astype(float)
        eta = np.empty((mx // 2, mz, my), dtype=np.complex64)
        phi = np.empty_like(eta)
        report = progress("Reading spectral planes", my)
        for j in range(my):
            plane = np.frombuffer(record((8 * mx * mz,)), dtype).reshape(mz, mx, 2)
            eta[:, :, j] = (plane[:, 0::2, 0] + 1j * plane[:, 1::2, 0]).T
            phi[:, :, j] = (plane[:, 0::2, 1] + 1j * plane[:, 1::2, 1]).T
            report(j + 1)
        # status='unknown' + rewind can leave stale bytes after the valid records.
    return meta, mean[:, 0], mean[:, 1], eta, phi


def to_host(a, xp):
    return a if xp is np else xp.asnumpy(a)


def synchronize(xp):
    if xp is not np:
        xp.cuda.Stream.null.synchronize()


class BatchedHelmholtz:
    """Same Chebyshev tau equations as the CPU script, without dense solves.

    Q integrates twice with zero constant/linear coefficients. Its even and
    odd blocks are tridiagonal. Two Thomas solves per parity determine the
    particular solution and response to its integration constant. A scalar
    boundary correction per parity enforces v(+/-1)=0.
    """
    def __init__(self, n, xp):
        self.n, self.xp = n, xp
        m = n - 2
        Q = ch.chebint(np.eye(m), m=2, axis=0)
        Q[:2] = 0
        low, diag, high = (np.zeros(n) for _ in range(3))
        for j in range(n):
            if 0 <= j - 2 < m:
                low[j] = Q[j, j - 2]
            if j < m:
                diag[j] = Q[j, j]
            if j + 2 < m:
                high[j] = Q[j, j + 2]
        self.low, self.diag, self.high = [xp.asarray(a) for a in (low, diag, high)]
        self.sums = xp.asarray(Q.sum(axis=0))

    def solve(self, q, phi):
        """q: (batch,), phi: (batch,n), output: (batch,n) complex128."""
        xp, n = self.xp, self.n
        m = n - 2
        count = len(q)
        # Stored mean mode has an independent solution; use a harmless q here.
        q = xp.where(q > 0, q, 1.0)
        b = xp.zeros((count, m), dtype=xp.complex128)
        response = xp.zeros((count, m), dtype=xp.float64)
        gamma = xp.zeros((count, m), dtype=xp.float64)
        constants = []
        for parity in (0, 1):
            indices = list(range(parity, m, 2))
            for j in indices:
                lower = -q * self.low[j]
                denominator = 1.0 - q * self.diag[j]
                if j > parity:
                    denominator = denominator - lower * gamma[:, j - 2]
                    rhs = phi[:, j] - lower * b[:, j - 2]
                    rr = -lower * response[:, j - 2]
                else:
                    rhs = phi[:, j]
                    rr = q
                gamma[:, j] = (-q * self.high[j]) / denominator
                b[:, j] = rhs / denominator
                response[:, j] = rr / denominator
            for j in reversed(indices[:-1]):
                b[:, j] -= gamma[:, j] * b[:, j + 2]
                response[:, j] -= gamma[:, j] * response[:, j + 2]
            s = self.sums[parity:m:2]
            constant = -(b[:, parity:m:2] * s).sum(axis=1) / (
                1.0 + (response[:, parity:m:2] * s).sum(axis=1))
            b[:, parity:m:2] += response[:, parity:m:2] * constant[:, None]
            constants.append(constant)
        v = xp.zeros((count, n), dtype=xp.complex128)
        v[:, :m] = b * self.diag[:m]
        v[:, 2:] += b * self.low[2:]
        v[:, :m - 2] += b[:, 2:] * self.high[:m - 2]
        v[:, 0] = constants[0]
        v[:, 1] = constants[1]
        return v


def velocity_coefficients(meta, u00, w00, eta, phi, xp=np, mode_batch=8192,
                          half_cheb_endpoints=False):
    nk, mz, n = eta.shape
    if phi.shape != eta.shape:
        raise ValueError("eta and phi must have the same shape")
    if mode_batch < 1:
        raise ValueError("mode_batch must be positive")
    solver = BatchedHelmholtz(n, xp)
    weights = np.ones(n)
    if half_cheb_endpoints:
        weights[[0, -1]] = 0.5
    wg = xp.asarray(weights)
    flat_eta, flat_phi = eta.reshape(-1, n), phi.reshape(-1, n)
    count = nk * mz
    outputs = [np.empty((count, n), dtype=np.complex64) for _ in range(3)]
    kz_modes = np.rint(np.fft.fftfreq(mz) * mz) * meta["bet"]
    report = progress("GPU mode reconstruction" if xp is not np else "CPU batched reconstruction", count)
    for start in range(0, count, mode_batch):
        stop = min(start + mode_batch, count)
        ids = np.arange(start, stop)
        kx = xp.asarray((ids // mz) * meta["alp"])
        kz = xp.asarray(kz_modes[ids % mz])
        q = kx * kx + kz * kz
        safe_q = xp.where(q > 0, q, 1.0)
        rhs = xp.asarray(flat_phi[start:stop], dtype=xp.complex128) * wg
        ec = xp.asarray(flat_eta[start:stop], dtype=xp.complex128) * wg
        vc = solver.solve(q, rhs)
        dv = xp.zeros_like(vc)
        dv[:, n - 2] = 2 * (n - 1) * vc[:, n - 1]
        for j in range(n - 3, -1, -1):
            dv[:, j] = dv[:, j + 2] + 2 * (j + 1) * vc[:, j + 1]
        dv[:, 0] *= 0.5
        uc = 1j * (kx[:, None] * dv - kz[:, None] * ec) / safe_q[:, None]
        wc = 1j * (kz[:, None] * dv + kx[:, None] * ec) / safe_q[:, None]
        for output, gpu in zip(outputs, (uc, vc, wc)):
            output[start:stop] = to_host(gpu.astype(xp.complex64), xp)
        report(stop)
    outputs[0][0] = u00 * weights
    outputs[1][0] = 0
    outputs[2][0] = w00 * weights
    return tuple(a.reshape(nk, mz, n) for a in outputs)


def write_velocity(meta, coefficients, out, xp=np, nx=None, nz=None,
                   y_batch=4, lab_frame=False, fft_double=False):
    """Evaluate Chebyshev series by batched matrix multiplication and GPU FFT.

    Keep only one complete spectral velocity component on the GPU at a time.
    Output axes are (x,y,z), with y decreasing from +1 to -1.
    """
    mx, my, mz = (meta[k] for k in ("mx", "my", "mz"))
    nk = mx // 2
    nx = int(np.ceil(1.5 * mx)) if nx is None else nx
    nz = int(np.ceil(1.5 * mz)) if nz is None else nz
    if nx < mx or nz < mz or y_batch < 1:
        raise ValueError("Require nx>=mx, nz>=mz and y_batch>=1")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shape = (nx, my, nz)
    print("Output grid {}: {:.3f} GiB in three float32 files".format(
        shape, 3 * nx * my * nz * 4 / 1024**3), flush=True)
    y = np.cos(np.pi * np.arange(my) / (my - 1))
    # cos(j*theta) gives T_j(y) directly at the Chebyshev nodes.
    T = np.cos(np.pi * np.arange(my)[:, None] * np.arange(my)[None, :] / (my - 1))
    dtype = xp.complex128 if fft_double else xp.complex64
    zindices = xp.asarray(np.rint(np.fft.fftfreq(mz) * mz).astype(int) % nz)
    for name, c in zip(("u", "v", "w"), coefficients):
        print("Uploading {} coefficients ({:.3f} GiB)".format(
            name, c.size * (16 if fft_double else 8) / 1024**3), flush=True)
        cg = xp.asarray(c.reshape(nk * mz, my), dtype=dtype)
        target = np.lib.format.open_memmap(out / (name + ".npy"), mode="w+",
                                          dtype="float32", shape=shape)
        report = progress("Transforming/writing " + name, my)
        for start in range(0, my, y_batch):
            stop = min(start + y_batch, my)
            basis = xp.asarray(T[start:stop].T, dtype=dtype)
            evaluated = (cg @ basis).T.reshape(stop - start, nk, mz)
            padded = xp.zeros((stop - start, nx // 2 + 1, nz), dtype=dtype)
            padded[:, :nk, zindices] = evaluated
            zx = xp.fft.ifft(padded, axis=2) * nz
            values = xp.fft.irfft(zx, n=nx, axis=1) * nx
            if name == "u" and lab_frame:
                values += meta["a0"]
            target[:, start:stop, :] = to_host(values.astype(xp.float32), xp).transpose(1, 0, 2)
            report(stop)
        print("Flushing " + name + ".npy", flush=True)
        target.flush()
        del target, cg, basis, evaluated, padded, zx, values
        if xp is not np:
            xp.get_default_memory_pool().free_all_blocks()
    np.save(out / "x.npy", np.arange(nx) * 2 * np.pi / (meta["alp"] * nx))
    np.save(out / "y.npy", y)
    np.save(out / "z.npy", np.arange(nz) * 2 * np.pi / (meta["bet"] * nz))
    metadata = dict(meta, shape=list(shape), axes=["x", "y", "z"],
                    lab_frame=lab_frame, fft_precision="double" if fft_double else "single",
                    backend="numpy" if xp is np else "cupy",
                    convention="normalized exp(+ikx*x+ikz*z), direct T_n coefficients")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def self_test(xp):
    """Check the selected backend against independent NumPy reference math."""
    rng = np.random.RandomState(42)
    n = 17
    q = np.array([0.0625, 1., 100., 1e4, 1e6])
    rhs = rng.randn(len(q), n) + 1j * rng.randn(len(q), n)
    actual = to_host(BatchedHelmholtz(n, xp).solve(xp.asarray(q), xp.asarray(rhs)), xp)
    Q = ch.chebint(np.eye(n - 2), m=2, axis=0)
    Q[:2] = 0
    B = np.column_stack((Q, np.eye(n, 2)))
    R = np.column_stack((np.eye(n - 2), np.zeros((n - 2, 2))))
    walls = np.vstack(((-1.) ** np.arange(n), np.ones(n))) @ B
    for i, wave in enumerate(q):
        A = np.vstack((R - wave * B[:n - 2], walls))
        b = np.concatenate((rhs[i, :n - 2], np.zeros(2)))
        reference = B @ np.linalg.solve(A, b)
        np.testing.assert_allclose(actual[i], reference, rtol=2e-7, atol=2e-9)
    c = rng.randn(7, n) + 1j * rng.randn(7, n)
    T = ch.chebvander(np.linspace(-1, 1, 4), n - 1).T
    got = to_host(xp.asarray(c, dtype=xp.complex64) @ xp.asarray(T, dtype=xp.complex64), xp)
    np.testing.assert_allclose(got, c @ T, rtol=1e-5, atol=1e-5)
    padded = (rng.randn(2, 7, 9) + 1j * rng.randn(2, 7, 9)).astype(np.complex64)
    pg = xp.asarray(padded)
    got = to_host(xp.fft.irfft(xp.fft.ifft(pg, axis=2) * 9, n=12, axis=1) * 12, xp)
    reference = np.fft.irfft(np.fft.ifft(padded, axis=2) * 9, n=12, axis=1) * 12
    np.testing.assert_allclose(got, reference, rtol=1e-4, atol=3e-5)
    print("PASS: batched Helmholtz solve, Chebyshev evaluation and inverse FFTs", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("field", nargs="?")
    parser.add_argument("--self-test", action="store_true", help="Test the selected backend without an input file")
    parser.add_argument("--out", default="velocity")
    parser.add_argument("--backend", choices=("cupy", "numpy"), default="cupy")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--mode-batch", type=int, default=8192, help="Fourier modes per solve batch")
    parser.add_argument("--y-batch", type=int, default=4, help="Wall-normal planes per FFT batch")
    parser.add_argument("--fft-double", action="store_true", help="Use double precision evaluation/FFTs")
    parser.add_argument("--nx", type=int)
    parser.add_argument("--nz", type=int)
    parser.add_argument("--endian", choices=(">", "<"), default=">")
    parser.add_argument("--lab-frame", action="store_true")
    parser.add_argument("--half-cheb-endpoints", action="store_true")
    args = parser.parse_args()
    if args.mode_batch < 1 or args.y_batch < 1:
        parser.error("Batch sizes must be positive")
    xp = np
    if args.backend == "cupy":
        try:
            import cupy as cp
        except ImportError:
            parser.error("CuPy is required for GPU mode. Install a CuPy build matching your CUDA setup.")
        xp = cp
        try:
            cp.cuda.Device(args.device).use()
            free, total = cp.cuda.runtime.memGetInfo()
            print("CUDA device {}: {:.2f}/{:.2f} GiB free".format(
                args.device, free / 1024**3, total / 1024**3), flush=True)
        except Exception as exc:
            parser.error("Cannot initialize CUDA: {}".format(exc))
    if args.self_test:
        self_test(xp)
        return
    if args.field is None:
        parser.error("An input field is required unless --self-test is used")
    started = monotonic()
    meta, u00, w00, eta, phi = read_field(args.field, args.endian)
    if (args.nx is not None and args.nx < meta["mx"]) or (args.nz is not None and args.nz < meta["mz"]):
        parser.error("Require nx>=mx and nz>=mz")
    t = monotonic()
    coeff = velocity_coefficients(meta, u00, w00, eta, phi, xp, args.mode_batch, args.half_cheb_endpoints)
    del eta, phi
    synchronize(xp)
    print("Reconstruction elapsed: {:.1f}s".format(monotonic() - t), flush=True)
    if xp is not np:
        xp.get_default_memory_pool().free_all_blocks()
    meta["half_cheb_endpoints"] = args.half_cheb_endpoints
    write_velocity(meta, coeff, args.out, xp, args.nx, args.nz, args.y_batch,
                   args.lab_frame, args.fft_double)
    print("Finished in {:.1f}s: {}".format(monotonic() - started, args.out), flush=True)


if __name__ == "__main__":
    main()

