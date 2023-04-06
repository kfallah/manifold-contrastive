"""
    Here we want to load up a Resnet model and construct
    a neareast neighbor graph from the activations of embedded
    CIFAR10 images.
"""
import sys
import random
import os
sys.path.append(os.path.dirname(os.getcwd()) + "/src/")

from tqdm import tqdm
import torch
import numpy as np
import torchvision
import torchvision.transforms.transforms as T

default_device = torch.device('cuda:0')
#backbone_load = torch.load("../results/SimCLR-Linear_09-13-2022_14-08-55/checkpoints/checkpoint_epoch299.pt")

"""
backbone = torchvision.models.resnet50(pretrained=True).to(default_device)
backbone.eval()
backbone.fc = torch.nn.Identity()
backbone.avgpool = torch.nn.Identity()
backbone.layer4 = torch.nn.Identity()
deactivate_requires_grad(backbone)
"""

def make_nn_graph(dataset, model, num_neighbors=5, batch_size=10):
    nn_graph = []
    # Make data loader
    # Go through and compute all of the embeddings
    embeddings = []
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    for batch in tqdm(dataloader):
        # Compute the embeddings
        embedding = model(batch.to(default_device))
        for index in range(len(embedding.shape[0])):
            embeddings.append(
                embedding[index].detach().cpu().numpy()
            )
    embeddings = torch.Tensor(embeddings)
    # Now go through and compute the nearest neighbors
    for index in tqdm(range(len(dataset))):
        embedding = embeddings[index]
        # Compute the distance to all other embeddings
        dists = torch.norm(embedding - embeddings, dim=1)
        # Argsort to get the nearest neighbors
        sorted_inds = torch.argsort(dists)
        # Add to the nn_graph
        nn_graph.append(
            sorted_inds[1:num_neighbors+1].detach().cpu().numpy()
        )
    return nn_graph

if __name__ == "__main__":
    # Load up the CIFAR10 dataset
    cifar10 = torchvision.datasets.CIFAR10(
        "../datasets",
        train=True,
        transform=T.Compose(
            [
                T.Resize(256),
                T.RandomCrop(224, 4),
                T.ToTensor(),
            ]
        ),
        download=True
    )
    # Load up the model
    model = torchvision.models.resnet18(pretrained=True).to(default_device)
    # Assign last layers to be identity operations in order to get the 
    # activations of the penultimate layer
    model.fc = torch.nn.Identity()
    model.avgpool = torch.nn.Identity()
    print(model)
    # Compute the nearest neighbors
    nn_graph = make_nn_graph(cifar10, model)
    # Save the graph 
    np.save("cifar10_resnet18_nn.npy", nn_graph)
