import torch

default_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def compute_operator_path_range(
    model,
    operator_index=0,
    device=default_device,
    num_samples=1000,
):
    """
        Computes the distribution of an operator's
        coefficients and returns the range of 5th to 95th 
        percentiles. 
    """
    # Get a bunch of random image samples
    # TODO fix this
    return (-3, 3)

def compute_operator_path_samples(
    initial_z,
    model,
    path_range=None,
    operator_index=0,
    num_samples=10,
    device=default_device,
):
    """
        Sample points along a manifold operator path
    """
    if path_range is None:
        path_range = compute_operator_path_range(
            model,
            operator_index=operator_index,
        )
    initial_z = initial_z.to(device)
    transop = model.contrastive_header.transop_header.transop
    # Get the operator
    psi = transop.get_psi()
    psi_operator = psi[:, operator_index]
    psi_operator = psi_operator.unsqueeze(0).to(device)
    num_coefficients = psi.shape[1]
    # Compute samples in operator time domain for given path range
    time_domain_samples = torch.linspace(
        path_range[0],
        path_range[1],
        num_samples,
    ).to(device)
    time_domain_samples = time_domain_samples[:, None, None, None]
    # Apply the operator transformation
    # NOTE: I am only applying a single operator, so I don't need to sum
    # Transform initial_z given the coefficients
    T = torch.matrix_exp(time_domain_samples * psi_operator)
    operator_path_samples = (
        T @ initial_z.reshape(1, 8, -1, 1)
    ).reshape(num_samples, -1)
    assert list(operator_path_samples.shape) == [num_samples, 512]

    return operator_path_samples

def compute_interpolation_dip_images(
    operator_path_samples,
    input_x,
    backbone,
    mse_lambda=1.0,
    learning_rate=1e-3,
    fixed_noise=False,
    return_network=False,
):
    """
        Computes the corresponding deep image priors for the given set of 
        operator path samples. 
    """

    return