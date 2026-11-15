import numpy as np
import torch
from src.model.preprocessing import ImagePreprocessor


def main():
    # Simulate a webcam frame (480 x 640)
    frame = np.random.randint(
        0, 256,
        (480, 640, 3),
        dtype=np.uint8
    )

    preprocessor = ImagePreprocessor()

    tensor = preprocessor.process(frame)

    print("Preprocessing test")
    print("-------------------")
    print("Input shape:", frame.shape)
    print("Output shape:", tensor.shape)
    print("Tensor type:", tensor.dtype)
    print("Tensor device:", tensor.device)
    print("Minimum value:", tensor.min().item())
    print("Maximum value:", tensor.max().item())

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32
    
    print("\nPreprocessing successful!")


if __name__ == "__main__":
    main()
