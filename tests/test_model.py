
import torch

from src.model.cnn_model import build_model, CLASS_NAMES


def main():
    model = build_model()

    # Evaluation mode
    model.eval()

    # Simulate a preprocessed image
    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    print("CNN Model Test")
    print("--------------")
    print("Classes:", CLASS_NAMES)
    print("Number of classes:", len(CLASS_NAMES))
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Backbone frozen:",
          not next(model.features.parameters()).requires_grad)
    print("Classifier trainable:",
          model.classifier[1].weight.requires_grad)

    assert output.shape == (1, len(CLASS_NAMES))
    assert not next(model.features.parameters()).requires_grad
    assert model.classifier[1].weight.requires_grad

    print("\nModel test successful!")


if __name__ == "__main__":
    main()
