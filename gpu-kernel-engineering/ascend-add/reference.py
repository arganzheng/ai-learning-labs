"""Ascend Add's fixed-shape contract and NumPy oracle; not an NPU simulator."""

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

SHAPE = (8, 2048)
CORE_COUNT = 8
BLOCK_LENGTH = 2048
TILE_LENGTH = 128
ELEMENT_COUNT = 16384


def fixture(case: str) -> tuple[NDArray[np.float16], NDArray[np.float16]]:
    indices = np.arange(ELEMENT_COUNT).reshape(SHAPE)
    if case == "signed":
        x = ((indices % 257 - 128) / 8).astype(np.float16)
        y = ((indices % 17 - 8) / 4).astype(np.float16)
    elif case == "zeros":
        x = np.zeros(SHAPE, dtype=np.float16)
        y = np.zeros(SHAPE, dtype=np.float16)
    elif case == "boundaries":
        x = np.zeros(SHAPE, dtype=np.float16)
        y = np.zeros(SHAPE, dtype=np.float16)
        x[:, ::TILE_LENGTH] = 2
        x[:, TILE_LENGTH - 1::TILE_LENGTH] = -3
        y[:, 0] = np.arange(CORE_COUNT, dtype=np.float16)
        y[:, -1] = -np.arange(CORE_COUNT, dtype=np.float16)
    else:
        raise ValueError(f"unknown fixture: {case}")
    return x, y


def tiled_reference(
    x: NDArray[np.float16], y: NDArray[np.float16]
) -> NDArray[np.float16]:
    for array in (x, y):
        if array.shape != SHAPE or array.dtype != np.float16:
            raise ValueError("the example supports only contiguous float16 (8, 2048)")
        if not array.flags.c_contiguous or not np.isfinite(array).all():
            raise ValueError("inputs must be contiguous and finite")
    result = np.empty(ELEMENT_COUNT, dtype=np.float16)
    coverage = np.zeros(ELEMENT_COUNT, dtype=np.int64)
    xf, yf = x.ravel(), y.ravel()
    for block in range(CORE_COUNT):
        for tile in range(BLOCK_LENGTH // TILE_LENGTH):
            start = block * BLOCK_LENGTH + tile * TILE_LENGTH
            end = start + TILE_LENGTH
            result[start:end] = xf[start:end] + yf[start:end]
            coverage[start:end] += 1
    np.testing.assert_array_equal(coverage, np.ones_like(coverage))
    return result.reshape(SHAPE)


def read_tensor(path: Path) -> NDArray[np.float16]:
    if path.stat().st_size != ELEMENT_COUNT * 2:
        raise ValueError(f"{path}: expected {ELEMENT_COUNT * 2} bytes")
    result = np.fromfile(path, dtype=np.float16).reshape(SHAPE)
    if not np.isfinite(result).all():
        raise ValueError(f"{path}: non-finite value")
    return result


def verify(x: Path, y: Path, output: Path) -> None:
    expected = (read_tensor(x).astype(np.float32) + read_tensor(y)).astype(np.float16)
    actual = read_tensor(output)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-5)
    print(f"PASS: every one of {ELEMENT_COUNT} values meets rtol=1e-3, atol=1e-5")


def audit() -> None:
    for case in ("signed", "zeros", "boundaries"):
        x, y = fixture(case)
        expected = (x.astype(np.float32) + y).astype(np.float16)
        np.testing.assert_array_equal(tiled_reference(x, y), expected)
        print(f"{case}: NumPy tiling covers {ELEMENT_COUNT} elements exactly once")
    x, y = fixture("signed")
    for invalid in (x.ravel(), x[:, ::-1]):
        try:
            tiled_reference(invalid, y)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input was accepted")
    print("unsupported shape/stride: rejected")
    print("block=2048, tile=128, iterations=16, queue_payload_bytes=1536")
    print("logical_global_bytes=98304, flops=16384, intensity=1/6")
    print("NumPy contract check only; no Ascend C compilation or execution")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit")
    generate = subparsers.add_parser("generate")
    generate.add_argument("--case", choices=("signed", "zeros", "boundaries"), default="signed")
    generate.add_argument("--root", type=Path, required=True)
    check = subparsers.add_parser("verify")
    check.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        audit()
    elif args.command == "generate":
        x, y = fixture(args.case)
        (args.root / "input").mkdir(parents=True, exist_ok=True)
        (args.root / "output").mkdir(parents=True, exist_ok=True)
        x.tofile(args.root / "input/input_x.bin")
        y.tofile(args.root / "input/input_y.bin")
        (x.astype(np.float32) + y).astype(np.float16).tofile(args.root / "output/golden.bin")
        print(f"Generated {args.case}; run the compiled kernel before verify")
    else:
        verify(args.root / "input/input_x.bin", args.root / "input/input_y.bin",
               args.root / "output/output_z.bin")


if __name__ == "__main__":
    main()
