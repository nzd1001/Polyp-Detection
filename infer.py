import torch
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

# Define argument parser
parser = argparse.ArgumentParser(description="Inference for image segmentation.")
parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint.")
parser.add_argument("--output_path", type=str, default="output.png", help="Path to save the output image.")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model.")
args = parser.parse_args()

# Define transformations for input image
def get_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

# Function to map class predictions to RGB colors
def decode_segmentation(segmentation, colormap):
    h, w = segmentation.shape
    decoded_img = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in colormap.items():
        decoded_img[segmentation == class_id] = color
    return decoded_img

# Define colormap (modify according to your classes)
COLORMAP = {
    0: [0, 0, 0],       # Background -> Black
    1: [255, 0, 0],     # Class 1 -> Red
    2: [0, 255, 0],     # Class 2 -> Green
    # Add more classes if needed
}

def main():
    # Load the model
    device = args.device
    model = torch.load(args.model_path, map_location=device)
    model.eval()  # Set model to evaluation mode

    # Load and preprocess the image
    image = Image.open(args.image_path).convert("RGB")
    transform = get_transform()
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)  # Output shape: [1, num_classes, H, W]
        prediction = torch.argmax(output.squeeze(0), dim=0).cpu().numpy()  # Shape: [H, W]

    # Decode segmentation mask to RGB
    segmented_image = decode_segmentation(prediction, COLORMAP)

    # Save and display the output
    output_image = Image.fromarray(segmented_image)
    output_image.save(args.output_path)
    print(f"Segmented image saved at: {args.output_path}")

    # Optionally, show the input and output side by side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Segmented Output")
    plt.imshow(segmented_image)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
