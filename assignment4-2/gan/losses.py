import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
from torch.autograd import Variable

def discriminator_loss(logits_real, logits_fake, device):
    """
    Computes the discriminator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    ones = Variable(torch.ones(logits_real.shape).to(device))
    zeros = Variable(torch.zeros(logits_fake.shape).to(device))
    loss = (bce_loss(logits_fake, zeros) + bce_loss(logits_real, ones))
    # loss /= logits_fake.shape[0]
    
    ##########       END      ##########
    
    return loss

def generator_loss(logits_fake, device):
    """
    Computes the generator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    ones = Variable(torch.ones(logits_fake.shape).to(device))
    loss = bce_loss(logits_fake, ones)
    # loss /= logits_fake.shape[0]
    ##########       END      ##########
    
    return loss


def ls_discriminator_loss(scores_real, scores_fake, device):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    ones = Variable(torch.ones(scores_real.shape).to(device))
    zeros = Variable(torch.zeros(scores_fake.shape).to(device))

    loss = .5 * torch.mean((scores_real - ones) ** 2) + .5 * torch.mean((scores_fake - zeros) ** 2)
    # loss /= scores_fake.shape[0]
    ##########       END      ##########
    
    return loss

def ls_generator_loss(scores_fake, device):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    ones = Variable(torch.ones(scores_fake.shape).to(device))

    loss = .5 * torch.mean((scores_fake - ones) ** 2)
    # loss /= scores_fake.shape[0]
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    
    
    ##########       END      ##########
    
    return loss
