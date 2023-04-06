import matplotlib.pyplot as plt
import torchvision.transforms.transforms as T

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