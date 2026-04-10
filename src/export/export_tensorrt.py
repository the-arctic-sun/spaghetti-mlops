import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT")

    parser.add_argument("--onnx", type=str, required=True,
                        help="Path to ONNX model")
    parser.add_argument("--engine", type=str, default="models/best.trt",
                        help="Output TensorRT engine path")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable FP16 optimization")

    return parser.parse_args()


def main():
    args = parse_args()

    onnx_path = Path(args.onnx)
    assert onnx_path.exists(), f"ONNX file not found: {onnx_path}"

    cmd = [
        "trtexec",
        f"--onnx={args.onnx}",
        f"--saveEngine={args.engine}",
        "--workspace=4096"
    ]

    if args.fp16:
        cmd.append("--fp16")

    print("Running:", " ".join(cmd))

    subprocess.run(cmd, check=True)

    print("TensorRT engine created")


if __name__ == "__main__":
    main()
