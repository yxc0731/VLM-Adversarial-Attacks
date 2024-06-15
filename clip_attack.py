from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
import pandas as pd

def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images
    
df = pd.read_csv('attack_dataset.csv')

model = CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-large-patch14")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

epsilon = 8/255  
alpha = 1/255

def optimize_image_with_noise(img_harm_path, img_adv_path, num_iterations=1000,
                              output_path="final_adv_image.png"):
    image_harm = Image.open(img_harm_path).convert("RGB")
    image_adv = Image.open(img_adv_path).convert("RGB")
    inputs_harm = processor(images=image_harm, return_tensors="pt").to(device)
    x = processor(images=image_adv, return_tensors="pt").pixel_values.to(device)

    adv_noise = torch.rand_like(x).to(device) * 2 * epsilon - epsilon
    x = denormalize(x).clone().to(device)
    adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
    adv_noise.requires_grad_(True)
    adv_noise.retain_grad()

    H_harm = model.get_image_features(**inputs_harm).detach()
                                  
    for iteration in range(num_iterations):
        adv_img = normalize(x + adv_noise)
        H_adv = model.get_image_features(pixel_values=adv_img)
        loss = torch.norm(H_adv - H_harm, p=2)
        loss.backward()

        adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(-epsilon, epsilon)
        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
        adv_noise.grad.zero_()
        if iteration % 100 == 0:
            print(iteration)
            print(loss)
        if loss.item() < 0.3:
            break

        if iteration % 1000 == 0:
            iter_image = to_pil_image((x + adv_noise).clamp(0, 1).detach().cpu().squeeze(0))
            iter_image.save(os.path.join(output_folder, f"adv_image_{iteration}.png"))


    final_adv_image = to_pil_image((x + adv_noise).clamp(0, 1).detach().cpu().squeeze(0))
    final_adv_image.save(output_path)

    return output_path


for index, row in df.iterrows():

    traget_toxic_img = row['traget_toxic_img']
    adv_toxic_img = row['adv_toxic_img']

    traget_toxic_img_path = f"/{traget_toxic_img}"
    adv_toxic_img_path = f"/{adv_toxic_img}"


    output_image_path = optimize_image(
        traget_toxic_img_path,
        "/clean.jpeg",
        learning_rate=0.01,
        num_iterations=1000,
        output_path=adv_toxic_img_path
    )

    print(f"Optimized image saved to: {output_image_path}")
