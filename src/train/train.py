import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO Segmentation Model")

    parser.add_argument("--model", type=str, default="yolo26n-seg.pt",
                        help="Pretrained model to start from")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--project", type=str, default="runs/segment",
                        help="Project directory")
    parser.add_argument("--name", type=str, default="train",
                        help="Experiment name")

    return parser.parse_args()


def main():
    args = parse_args()

    print("Starting Training with config:")
    print(vars(args))

    # Load model
    model = YOLO(args.model)

    # Train
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name
    )

    print("Training completed!")


if __name__ == "__main__":
    main()
