"""
    Here we want to train an inverse function that maps
    from the final layer of the backbone model to the 
    layer before using a deconvolution layer. 
"""
import torch
import argparse
import os
import wandb
from tqdm import tqdm

from lightly.models.utils import deactivate_requires_grad
import torchvision
import torchvision.transforms.transforms as T

default_device = torch.device("cuda:0")

def load_resnet_backbone(backbone_name="resnet18", last_layer="fc"):
    if backbone_name == "resnet18":
        backbone = torchvision.models.resnet18(pretrained=True).to(default_device)
    else:
        backbone = torchvision.models.resnet50(pretrained=True).to(default_device)
    backbone.eval()

    if last_layer == "layer4":
        backbone.avgpool = torch.nn.Identity()
        backbone.layer4 = torch.nn.Identity()
    backbone.fc = torch.nn.Identity()

    deactivate_requires_grad(backbone)

    return backbone

def make_network(input_shape=(1, 2048), output_shape=(1, 2048, 7, 7)):
    # NOTE: Initially just using a single linear layer to invert the 
    # global pooling operation. 
    model = torch.nn.Sequential(
        torch.nn.Linear(input_shape[0], output_shape[0]),
        torch.nn.ReLU(),
        torch.nn.Unflatten(1, output_shape[1:])
    )

    return model

def make_training_dataset(
    image_dataset,
    backbone_input_layer,
    backbone_output_layer,
    save_path="data/single_layer_inverse_dataset.pt",
    num_samples=1000,
):
    if os.path.exists(save_path):
        dataset = torch.load(save_path)
        return dataset
    else:
        dataset = []
        for example_index in tqdm(range(num_samples)):
            # Get a random input image
            print(image_dataset[example_index])
            image, _ = image_dataset[example_index]
            image = image.cuda()
            # Get the output of the backbone
            backbone_output = backbone_output_layer(image)
            backbone_input = backbone_input_layer(image)

            dataset.append(
                (backbone_output, backbone_input)
            )
        # Save the dataset
        torch.save(dataset, save_path)
        return dataset

def evaluate_model(model, eval_dataset):
    model.eval()
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=32,
        shuffle=False,
    )
    losses = []
    for batch_output, batch_input in eval_dataloader:
        batch_output = batch_output.to(default_device)
        batch_input = batch_input.to(default_device)
        # Do a forward pass given the output to produce the input
        estimate_input = model(batch_output)
        # Compute the loss
        loss = torch.nn.functional.mse_loss(estimate_input, batch_input)
        losses.append(loss.item())

    mean_loss = sum(losses) / len(losses)
    return mean_loss

def train_model(
    model,
    train_dataset, 
    eval_dataset,
    num_epochs=10,
    batch_size=32,
    args=None,
):
    wandb.init(
        project="deep_image_prior",
        group=args.wandb_group_name,
    )
    # Make the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # Write training loop
    for epoch in tqdm(range(num_epochs)):
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        losses = []

        for batch_output, batch_input in train_dataloader:
            batch_output = batch_output.to(default_device)
            batch_input = batch_input.to(default_device)
            # Do a forward pass given the output to produce the input
            estimate_input = model(batch_output)
            # Compute the loss
            loss = torch.nn.functional.mse_loss(estimate_input, batch_input)
            losses.append(loss.item())
            # Backprop
            loss.backward()
            # Update the weights
            optimizer.step()
            # Zero the gradients
            optimizer.zero_grad()

        mean_loss = sum(losses) / len(losses)

        # Do evaluation
        test_loss = evaluate_model(
            model, 
            eval_dataset
        )

        wandb.log({
            "epoch": epoch,
            "train_loss": mean_loss,
            "test_loss": test_loss
        })

if __name__ == "__main__":
    # Setup argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--backbone_name", default="resnet18")
    parser.add_argument("--last_layer", default="fc")
    parser.add_argument("--learning_rate", default=1e-3)
    parser.add_argument("--num_epochs", default=2)
    parser.add_argument("--wandb_group_name", default="single_layer_inverse")

    args = parser.parse_args()

    model = make_network()
    # Load cifar 10
    train_cifar10 = torchvision.datasets.CIFAR10(
        "../../datasets",
        train=True,
        transform=T.Compose([
            T.Resize(256),
            T.RandomCrop(224, 4),
            T.ToTensor(),
        ]),
        download=True
    )
    test_cifar10 = torchvision.datasets.CIFAR10(
        "../../datasets",
        train=False,
        transform=T.Compose([
            T.Resize(256),
            T.RandomCrop(224, 4),
            T.ToTensor(),
        ]),
        download=True
    )
    # Load the backbone networks
    backbone_input_layer = load_resnet_backbone(
        args.backbone_name,
        last_layer="fc",
    )
    backbone_output_layer = load_resnet_backbone(
        args.backbone_name,
        last_layer="layer4",
    )
    print("Making train dataset")
    train_dataset = make_training_dataset(
        train_cifar10,
        backbone_input_layer=backbone_input_layer,
        backbone_output_layer=backbone_output_layer,
        save_path="data/single_layer_inverse_train_dataset.pt",
    )
    print("Making test dataset")
    test_dataset = make_training_dataset(
        test_cifar10,
        save_path="data/single_layer_inverse_test_dataset.pt",
    )
    print("Training model")
    train_model(
        model,
        train_dataset,
        test_dataset,
        num_epochs=args.num_epochs,
        args=args
    )