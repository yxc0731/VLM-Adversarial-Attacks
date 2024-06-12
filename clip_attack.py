from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image


model = CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-large-patch14")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def optimize_image(img_harm_path, img_adv_path, learning_rate=0.01, num_iterations=1000, output_path="adv_image.png"):

    image_harm = Image.open(img_harm_path).convert("RGB")
    image_adv = Image.open(img_adv_path).convert("RGB")


    inputs_harm = processor(images=image_harm, return_tensors="pt").to(device)
    inputs_adv = processor(images=image_adv, return_tensors="pt").to(device)


    H_harm = model.get_image_features(**inputs_harm).detach()


    x_adv = inputs_adv.pixel_values.requires_grad_(True)


    optimizer = torch.optim.Adam([x_adv], lr=learning_rate)


    for iteration in range(num_iterations):
        print(iteration)
        optimizer.zero_grad()
        H_adv = model.get_image_features(pixel_values=x_adv)
        loss = torch.norm(H_adv - H_harm, p=2)
        print(loss)
        loss.backward()
        optimizer.step()

        if loss.item() < 0.3:
            break


    final_adv_image = to_pil_image(torch.clamp(x_adv.detach().cpu().squeeze(0), 0, 1))

    final_adv_image.save(output_path)

    return output_path


output_image_path = optimize_image(
    "/Illegal_activity/arson.png",
    "/Illegal_activity_adv/arson.png",
    learning_rate=0.01,
    num_iterations=1000,
    output_path="/clip/path_to_output_image.png"
)

print(f"Optimized image saved to: {output_image_path}")
