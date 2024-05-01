from facenet_pytorch import MTCNN, InceptionResnetV1, alignment
from PIL import Image, ImageDraw
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
image_path = 'fan_images/test_images/'
image_names = ["9_0.jpg", "9.jpg", "9_3.jpg"]
image_save_path = 'fan_images/corpped/'
post_process = False
keep_all = True
if len(image_names) > 1:
    imgs = [Image.open(image_path + n).convert("RGB") for n in image_names]
else:
    imgs = Image.open(image_path + image_names[0]).convert("RGB")
mtcnn = MTCNN(device='cpu', post_process=post_process, margin=10, keep_all=keep_all, align=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cpu')
imgs_cpd, box = mtcnn(imgs, return_box=True)
aligned = []
for heads in imgs_cpd:
    if heads is not None:
        aligned.append(heads)
aligned = torch.cat(aligned, axis=0)
embeddings = resnet(aligned).detach().cpu()
centroid = embeddings.mean(dim=0)
x = (embeddings - centroid).norm(dim=1).cpu()
radius = (embeddings - centroid).norm(dim=1).max().cpu()
[[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
