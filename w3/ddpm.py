# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm
from unet import Unet
from fid import compute_fid


class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)
    
    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The negative ELBO of the batch of dimension `(batch_size,)`.
        """

        ### Implement Algorithm 1 here ###

        eps = torch.normal(0, 1, size=x.size()).to(x.device)
        t = torch.randint(0, self.T, size=(x.shape[0],)).to(x.device)
        term1 = torch.sqrt(self.alpha_cumprod[t.long()])
        term2 = torch.sqrt(1-self.alpha_cumprod[t.long()])
        z = term1[:, None] * x + term2[:, None] * eps
        t.unsqueeze_(-1)
        eps_theta = self.network(z, t)
        neg_elbo = torch.norm(eps - eps_theta, dim=tuple(range(1, eps.dim())))
        return neg_elbo

    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        # Sample x_t for t=T (i.e., Gaussian noise)
        x_t = torch.randn(shape).to(self.alpha.device)

        # Sample x_t given x_{t+1} until x_0 is sampled

        for t in range(self.T-1, -1, -1):
            ### Implement the remaining of Algorithm 2 here ###
            if t>0:
                z = torch.randn_like(x_t)
            else:
                z = torch.zeros_like(x_t)
                
            #TypeError: expected Tensor as element 0 in argument 0, but got int
            eps_theta = self.network(x_t, torch.full((x_t.shape[0], 1), t).to(self.alpha.device))

            x_t = 1/torch.sqrt(self.alpha[t]) * (x_t - (1-self.alpha[t])/torch.sqrt(1-self.alpha_cumprod[t]) * eps_theta + z * torch.sqrt(self.beta[t]))
        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for batch_idx, x in enumerate(data_iter):
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            
            # Check gradient flow
            max_grad = max([p.grad.abs().max().item() for p in model.parameters() if p.grad is not None])
            
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}", max_grad=f"{max_grad:.2e}")
            progress_bar.update()


class FcNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden, T):
        """
        Initialize a fully connected network for the DDPM, where the forward function also take time as an argument.
        
        parameters:
        input_dim: [int]
            The dimension of the input data.
        num_hidden: [int]
            The number of hidden units in the network.
        """
        super(FcNetwork, self).__init__()
        self.T = T
        self.network = nn.Sequential(nn.Linear(input_dim+1, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, input_dim))

    def forward(self, x, t):
        """"
        Forward function for the network.
        
        parameters:
        x: [torch.Tensor]
            The input data of dimension `(batch_size, input_dim)`
        t: [torch.Tensor]
            The time steps to use for the forward pass of dimension `(batch_size, 1)`
        """
        t_normalized = t.float() / self.T
        x_t_cat = torch.cat([x, t_normalized], dim=1)
        return self.network(x_t_cat)


if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    import ToyData

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test', 'fid'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='tg', choices=['tg', 'cb', 'mnist'], help='dataset to use {tg: two Gaussians, cb: chequerboard} (default: %(default)s)')
    parser.add_argument('--network', type=str, default='unet', choices=['unet', 'fcnetwork'], help='network architecture to use {unet: convolutional U-Net, fcnetwork: fully connected} (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--classifier', type=str, default='mnist_classifier.pth', help='path to classifier checkpoint for FID evaluation (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    # Generate the data
    n_data = 10000000
    toy = {'tg': ToyData.TwoGaussians, 'cb': ToyData.Chequerboard}[args.data]()
    transform = lambda x: (x-0.5)*2.0
    train_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)

    
    import torch
    from torchvision import datasets, transforms

    # Define the transform pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add slight noise (dequantization)
        transforms.Lambda(lambda x: x + torch.rand(x.shape) / 255),
        # Rescale from [0, 1] to [-1, 1]
        transforms.Lambda(lambda x: (x - 0.5) * 2.0),
        # Flatten the image into a 1D vector (784 for MNIST)
        transforms.Lambda(lambda x: x.flatten())
    ])

    # Load the dataset
    mnist_train_data = datasets.MNIST(
        root='data/',
        train=True,
        download=True,
        transform=transform
    )
    mnist_test_data = datasets.MNIST(
        root='data/',
        train=False,
        download=True,
        transform=transform
    )
    
    mnist_train_loader = torch.utils.data.DataLoader(mnist_train_data, batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(mnist_test_data, batch_size=args.batch_size, shuffle=False)
    
    
    # Get the dimension of the dataset
    D = next(iter(mnist_train_loader))[0].shape[1]

    # Set the number of steps in the diffusion process
    T = 1000
    
    # Define the network based on CLI argument
    if args.network == 'unet':
        network = Unet()
    else:  # fcnetwork
        num_hidden = 64
        network = FcNetwork(D, num_hidden, T=T)


    # Define model
    model = DDPM(network, T=T).to(args.device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        import matplotlib.pyplot as plt
        import numpy as np

        # Load the model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = model.sample((64, D)).cpu() 

        # Reverse transformation: from [-1, 1] back to [0, 1]
        samples = samples / 2.0 + 0.5
        
        # Reshape from flattened (784) to images (28, 28)
        samples = samples.reshape(-1, 28, 28)
        
        # Clip to valid range
        samples = torch.clamp(samples, 0, 1)

        # Plot the samples in a grid
        fig, axes = plt.subplots(8, 8, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            ax.imshow(samples[i], cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(args.samples)
        plt.close()

    elif args.mode == 'fid':
        import matplotlib.pyplot as plt
        import numpy as np
        from torchvision import datasets, transforms

        # Load the model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()

        # Generate samples
        with torch.no_grad():
            samples = model.sample((1000, D)).cpu() 

        # Reverse transformation: from [-1, 1] back to [0, 1]
        samples = samples / 2.0 + 0.5
        
        # Reshape from flattened (784) to images (1, 28, 28)
        samples = samples.reshape(-1, 1, 28, 28)
        
        # Clip to valid range
        samples = torch.clamp(samples, 0, 1)
        
        # Rescale back to [-1, 1] for FID computation
        samples_fid = (samples - 0.5) * 2.0

        # Load real test data with proper transform for FID
        transform_fid = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.rand(x.shape) / 255),
            transforms.Lambda(lambda x: (x - 0.5) * 2.0),
        ])
        
        mnist_test_data = datasets.MNIST(
            root='data/',
            train=False,
            download=True,
            transform=transform_fid
        )
        
        # Get first 1000 real test samples
        real_samples = []
        for i in range(min(1000, len(mnist_test_data))):
            real_samples.append(mnist_test_data[i][0])
        real_samples = torch.stack(real_samples).to(args.device)

        # Move generated samples to device
        samples_fid = samples_fid.to(args.device)

        # Compute FID
        fid_score = compute_fid(
            real_samples,
            samples_fid,
            device=args.device,
            classifier_ckpt=args.classifier
        )
        
        print(f"\nFID Score: {fid_score:.4f}")
