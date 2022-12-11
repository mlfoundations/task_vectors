import os
import torch
from tqdm import tqdm

import open_clip

from src.datasets.templates import get_templates
from src.datasets.registry import get_dataset

from src.modeling import ClassificationHead, ImageEncoder


def build_classification_head(model, dataset_name, template, data_location, device):
    template = get_templates(dataset_name)

    logit_scale = model.logit_scale
    dataset = get_dataset(
        dataset_name,
        None,
        location=data_location
    )
    model.eval()
    model.to(device)

    print('Building classification head.')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(dataset.classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = open_clip.tokenize(texts).to(device) # tokenize
            embeddings = model.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()
        
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head


def get_classification_head(args, dataset):
    filename = os.path.join(args.save, f'head_{dataset}.pt')
    if os.path.exists(filename):
        print(f'Classification head for {args.model} on {dataset} exists at {filename}')
        return ClassificationHead.load(filename)
    print(f'Did not find classification head for {args.model} on {dataset} at {filename}, building one from scratch.')
    model = ImageEncoder(args, keep_lang=True).model
    template = get_templates(dataset)
    classification_head = build_classification_head(model, dataset, template, args.data_location, args.device)
    os.makedirs(args.save, exist_ok=True)
    classification_head.save(filename)
    return classification_head

