import argparse
import torch
from torchvision import transforms
from PIL import Image
import os
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from torchvision.transforms import Resize, ToTensor, ToPILImage, Compose, InterpolationMode,Normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_args():
    parser = argparse.ArgumentParser(description="Run inference on an image using a segmentation model")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    return parser.parse_args()

# Load model
def load_model():
    model=smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3   
)
    checkpoint=torch.load("model/model.pth",weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model=model.to(device)
    model.eval()  
    return model
def preprocess_image(image_path):
    img=Image.open(image_path)
    transform = Compose([Resize((256,256),interpolation=InterpolationMode.BILINEAR),
                   ToTensor(),Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    h=img.size[1]
    w=img.size[0]
    data=transform(img)
    return data.unsqueeze(0).to(device),h,w
def main():
    args = get_args()
    model = load_model()
    data,h,w = preprocess_image(args.image_path)
    with torch.no_grad():
        mask=model(data)
        mask=mask.squeeze(0).cpu()
        one_hot_mask=(F.one_hot(torch.argmax(mask, 0)).permute(2,0,1).float()) #convert mask to image
        one_hot_mask[2,:,:]=0 #convert blue to black
        mask2image=ToPILImage()(one_hot_mask)
        final_image=Resize((h, w), interpolation=InterpolationMode.NEAREST)(mask2image) #resize to original size
        final_image.save("./output_mask.png")

if __name__ == '__main__':
    main()
