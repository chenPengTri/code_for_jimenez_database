"""Read the supplied Fortran format and reconstruct channel velocity.

Requires NumPy and SciPy. Assumptions:
  f = sum c[p,k,n] T_n(y) exp(+i*(p*alp*x + k*bet*z)).
  Stored Fourier coefficients are normalized series coefficients.
  Stored Chebyshev coefficients multiply T_n directly (no half weights).
  omega_y = du/dz - dw/dx; channel walls are y = +/-1.
  z modes are [0, 1, ..., K, -K, ..., -1], with odd mz.
No file from the actual database has been used to validate these conventions.
"""
import argparse
import json
import struct
from time import monotonic
from pathlib import Path

import numpy as np
from numpy.polynomial import chebyshev as ch
from scipy import sparse
from scipy.sparse.linalg import splu


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


class ChebyshevHelmholtz:
    """Sparse Chebyshev tau solve: (d2/dy2 - q)v = phi, v(+/-1)=0.

    Represent v = Q*b + c0 + c1*y, where Q integrates twice and b=v''.
    Match the lowest n-2 equation coefficients and both wall conditions.
    """
    def __init__(self, n):
        self.n = n
        Q = ch.chebint(np.eye(n - 2), m=2, axis=0)
        Q[:2] = 0.0  # Integration constants are independent unknowns.
        E = np.eye(n, 2)
        self.B = sparse.csc_matrix(np.column_stack((Q, E)))
        self.R = sparse.hstack((sparse.eye(n - 2), sparse.csc_matrix((n - 2, 2))), format="csc")
        wall_eval = np.vstack(((-1.0) ** np.arange(n), np.ones(n)))
        self.walls = sparse.csc_matrix(wall_eval @ self.B)

    def solve(self, q, rhs):
        if q <= 0:
            raise ValueError("The mean mode is handled separately")
        A = sparse.vstack((self.R - q * self.B[:self.n - 2], self.walls), format="csc")
        b = np.zeros((self.n,) + rhs.shape[1:], dtype=np.complex128)
        b[:self.n - 2] = rhs[:self.n - 2]
        # A is real; solve both real and imaginary parts with one factorization.
        lu = splu(A)
        solution = lu.solve(b.real) + 1j * lu.solve(b.imag)
        return self.B @ solution


def velocity_coefficients(meta, u00, w00, eta, phi, half_cheb_endpoints=False):
    """Return (uhat, vhat, what) in Fourier-Chebyshev coefficient space.

    If half_cheb_endpoints=True, divide first and last Chebyshev coefficients
    by two before use (only select this for an endpoint-weighted convention).
    """
    nk, mz, n = eta.shape
    if phi.shape != eta.shape or (2 * nk, n, mz) != tuple(meta[k] for k in ("mx", "my", "mz")):
        raise ValueError("Coefficient shapes do not match metadata")
    weights = np.ones(n)
    if half_cheb_endpoints:
        weights[[0, -1]] = 0.5
    uhat, vhat, what = (np.zeros_like(eta) for _ in range(3))
    solver = ChebyshevHelmholtz(n)
    kz = np.rint(np.fft.fftfreq(mz) * mz) * meta["bet"]
    groups = mz // 2 + 1
    report = progress("Reconstructing Fourier mode groups", nk * groups)
    for p in range(nk):
        kx = p * meta["alp"]
        # +/- kz share one Helmholtz factorization.
        for k in range(mz // 2 + 1):
            indices = [k] if k == 0 else [k, mz - k]
            q = kx**2 + kz[k]**2
            if q == 0:
                report(p * groups + k + 1)
                continue
            rhs = (phi[p, indices, :] * weights).T
            vc = solver.solve(q, rhs)
            dvc = ch.chebder(vc, axis=0)
            dv = np.zeros_like(vc)
            dv[:n - 1] = dvc
            ec = (eta[p, indices, :] * weights).T
            uhat[p, indices, :] = (1j * (kx * dv - kz[indices] * ec) / q).T
            vhat[p, indices, :] = vc.T
            what[p, indices, :] = (1j * (kz[indices] * dv + kx * ec) / q).T
            report(p * groups + k + 1)
    # The mean horizontal velocities cannot be recovered from eta and phi.
    uhat[0, 0] = u00 * weights
    what[0, 0] = w00 * weights
    # Mean wall-normal velocity is zero for an impermeable channel.
    return uhat, vhat, what


def velocity_planes(meta, coefficients, nx=None, nz=None, lab_frame=False):
    """Yield (y, u, v, w); each velocity plane has shape (nx, nz).

    Default x,z grids use 3/2 padding. y runs from +1 to -1.
    lab_frame=True adds a0 to u, assuming U_lab = U_stored + a0.
    """
    mx, my, mz = (meta[n] for n in ("mx", "my", "mz"))
    nx = int(np.ceil(1.5 * mx)) if nx is None else nx
    nz = int(np.ceil(1.5 * mz)) if nz is None else nz
    if nx < mx or nz < mz:
        raise ValueError("Require nx >= mx and nz >= mz to retain all modes")
    nk = mx // 2
    zmodes = np.rint(np.fft.fftfreq(mz) * mz).astype(int)
    zindices = zmodes % nz
    for y in np.cos(np.pi * np.arange(my) / (my - 1)):
        fields = []
        for c in coefficients:
            # chebval expects the polynomial coefficient index first.
            fhat = ch.chebval(y, np.moveaxis(c, -1, 0))
            padded = np.zeros((nx // 2 + 1, nz), dtype=np.complex128)
            padded[:nk, zindices] = fhat
            # Inverse z first; irfft in x supplies the omitted conjugate modes.
            # Default inverse FFTs divide by the transformed axis length.
            # Undo that scaling explicitly for compatibility with older NumPy.
            zx = np.fft.ifft(padded, axis=1) * nz
            fields.append(np.fft.irfft(zx, n=nx, axis=0) * nx)
        if lab_frame:
            fields[0] += meta["a0"]
        yield (y, fields[0], fields[1], fields[2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("field")
    parser.add_argument("--out", default="velocity")
    parser.add_argument("--endian", choices=(">", "<"), default=">")
    parser.add_argument("--nx", type=int)
    parser.add_argument("--nz", type=int)
    parser.add_argument("--lab-frame", action="store_true")
    parser.add_argument("--half-cheb-endpoints", action="store_true")
    args = parser.parse_args()
    started = monotonic()
    meta, u00, w00, eta, phi = read_field(args.field, args.endian)
    coeff = velocity_coefficients(meta, u00, w00, eta, phi, args.half_cheb_endpoints)
    del eta, phi
    nx = args.nx if args.nx is not None else int(np.ceil(1.5 * meta["mx"]))
    nz = args.nz if args.nz is not None else int(np.ceil(1.5 * meta["mz"]))
    if nx < meta["mx"] or nz < meta["mz"]:
        raise ValueError("Require nx >= mx and nz >= mz")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shape = (nx, meta["my"], nz)
    print(f"Output grid (nx, ny, nz) = {shape}; "
          f"velocity files total {3 * np.prod(shape) * 4 / 1024**3:.3f} GiB", flush=True)
    arrays = [np.lib.format.open_memmap(out / f"{name}.npy", mode="w+", dtype="float32", shape=shape)
              for name in ("u", "v", "w")]
    ys = []
    report = progress("Transforming and writing velocity planes", meta["my"])
    for j, (y, *planes) in enumerate(velocity_planes(meta, coeff, nx, nz, args.lab_frame)):
        ys.append(y)
        for target, plane in zip(arrays, planes):
            target[:, j, :] = plane
        report(j + 1)
    print("Flushing velocity files and saving coordinates...", flush=True)
    for target in arrays:
        target.flush()
    np.save(out / "x.npy", np.arange(nx) * (2 * np.pi / meta["alp"] / nx))
    np.save(out / "y.npy", np.array(ys))
    np.save(out / "z.npy", np.arange(nz) * (2 * np.pi / meta["bet"] / nz))
    meta.update(shape=list(shape), axes=["x", "y", "z"], lab_frame=args.lab_frame,
                convention="exp(+ikx*x+ikz*z), normalized Fourier series, direct T_n coefficients",
                half_cheb_endpoints=args.half_cheb_endpoints)
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Saved u, v, w and coordinates to {out}; shape = {shape}. "
          f"Total elapsed: {monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
