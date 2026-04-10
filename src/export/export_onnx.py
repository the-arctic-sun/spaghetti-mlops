import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX")

    parser.add_argument("--model", type=str, required=True,
                        help="Path to .pt model")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size")
    parser.add_argument("--opset", type=int, default=12,
                        help="ONNX opset version")

    return parser.parse_args()


def main():
    args = parse_args()

    print(" Exporting to ONNX with config:")
    print(vars(args))

    model = YOLO(args.model)

    model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=False   # IMPORTANT for TensorRT
    )

    print(" ONNX export completed!")


if __name__ == "__main__":
    main()
