import argparse
from ultralytics import YOLO
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Segmentation Inference")

    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model (.pt)")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to image or folder")
    parser.add_argument("--output", type=str, default="outputs/predict",
                        help="Output directory")
    parser.add_argument("--save", action="store_true",
                        help="Save prediction images")

    return parser.parse_args()


def main():
    args = parse_args()

    print("Running inference with config:")
    print(vars(args))

    # Load model
    model = YOLO(args.model)

    # Run inference
    results = model.predict(
        source=args.source,
        save=args.save,
        project=args.output,
        name="run"
    )

    print(f" Inference completed. Results saved to: {args.output}")

    # Optional: print summary
    for r in results:
        if r.boxes:
            print("Detected classes:", r.boxes.cls.tolist())


if __name__ == "__main__":
    main()
