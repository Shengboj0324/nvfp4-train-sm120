"""Exact bias of 8-bit software SR onto E2M1: enumerate all 256 random bytes per input value."""
import numpy as np

GRID = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)


def step_of(a):
    return np.where(a >= 4, 2.0, np.where(a >= 2, 1.0, 0.5)).astype(np.float32)


def int_sr(x, r8, half_offset=True):
    a = np.abs(x).astype(np.float32)
    off = np.where(a < 1, np.float32(1), np.float32(0)).astype(np.float32)
    z = (a + off).astype(np.float32)
    b = z.view(np.uint32).astype(np.uint64) + (r8.astype(np.uint64) << 14) + (np.uint64(1 << 13) if half_offset else 0)
    b = (b & 0xFFC00000).astype(np.uint32)
    q = np.minimum(b.view(np.float32) - off, np.float32(6))
    return np.copysign(q, x)


def te_sr(x, r8):
    u = (r8 & 0xFF).astype(np.float32) * np.float32(1 / 256)
    a = np.abs(x).astype(np.float32)
    t = (u * step_of(a) + a).astype(np.float32)
    st = step_of(t)
    q = np.minimum(np.floor(t / st) * st, 6)
    return np.copysign(q, x)


def exact_sr_mean(x):
    return x  # ideal SR is unbiased inside [-6, 6]


rng = np.random.default_rng(0)
x = rng.uniform(-6, 6, 200000).astype(np.float32)
r = np.arange(256, dtype=np.uint32)
for name, fn in [("int_sr SRF (+half LSB)", lambda xx, rr: int_sr(xx, rr, True)),
                 ("int_sr SRFF (no offset)", lambda xx, rr: int_sr(xx, rr, False)),
                 ("TE software (SRFF)", te_sr)]:
    q = fn(x[:, None], r[None, :])
    on_grid = np.isin(np.abs(q), GRID).all()
    mean = q.mean(axis=1)
    err = (mean - x) / step_of(np.abs(x))  # bias in units of the local grid step
    # signed bias relative to |x| direction (toward zero = negative)
    signed = err * np.sign(x)
    print(f"{name:26s} on_grid={on_grid}  mean bias toward |x| = {signed.mean():+.6f} step  "
          f"max |bias| = {np.abs(err).max():.6f} step  (2^-9 = {2**-9:.6f})")
