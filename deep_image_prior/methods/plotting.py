import matplotlib.pyplot as plt
import torchvision.transforms.transforms as T

from methods.manifold_interpolation import compute_operator_path_range, compute_operator_path_samples
from methods.deep_image_prior import compute_dip_image

def plot_multiple_dip_images(
    original_images,
    dip_images,
    save_path="dip_comparisons.png",
    image_shape=(32, 32)
):
    fig, axs = plt.subplots(len(original_images), 2, figsize=(len(original_images), 2))
    for index in range(len(original_images)):
        original_image = original_images[index]
        dip_image = dip_images[index]
        # Resize images to the proper shape
        dip_image = T.Resize(image_shape)(dip_image)
        original_image = T.Resize(image_shape)(original_image)
        # dip_image = np.resize(dip_image, image_shape)
        # original_image = np.resize(original_image, image_shape)
        # Convert to numpy and permute the channels
        dip_image = dip_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        original_image = original_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        # Display the images
        axs[index, 0].imshow(original_image)
        axs[index, 0].axis('off')
        axs[index, 1].imshow(dip_image)
        axs[index, 1].axis('off')

    plt.savefig(save_path)

def plot_operator_path_samples(
    input_images,
    dip_images,
    save_path="nn_path_visualizations.png",
):
    """
        Plots the nearest images to the points along a
        manifold path for a given transport operator.
    """
    fig, axs = plt.subplots(
        len(dip_images),
        len(dip_images[0]) + 1,
        figsize=((len(dip_images[0]) + 1) * 1.5, len(input_images)),
        dpi=300
    )
    for input_image_index in range(len(input_images)):
        input_image = input_images[input_image_index]
        axs[z_index, 0].imshow(input_image)
        # axs[index].set_title(f"Path {index}")
        axs[z_index, 0].axis("off")
        axs[z_index, 0].set_title("Initial Image")
        # Plot the images
        # axs = image_fig.subplots(1, num_samples)
        for index in range(num_samples):
            # print(images[index].shape)
            image = dip_images[input_image_index, index].permute(1, 2, 0) # [:, :, [2, 1, 0]]
            image = (image - image.min()) / (image.max() - image.min())
            axs[z_index, index + 1].imshow(image)
            # axs[index].set_title(f"Path {index}")
            axs[z_index, index + 1].axis("off")

    plt.savefig(save_path)