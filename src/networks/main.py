from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .cifar10_LeNet_elu import CIFAR10_LeNet_ELU, CIFAR10_LeNet_ELU_Autoencoder
from .cycif_Vanilla_AE import CyCIF_VanillaNetwork, CyCIF_VanillaAE
from .cycif_ResNet import CyCIF_ResidualNetwork, CyCIF_ResNetAE

def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'cycif_Net', 'cycif_ResNet')
    assert net_name in implemented_networks

    net = None

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    if net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()

    if net_name == 'cifar10_LeNet_ELU':
        net = CIFAR10_LeNet_ELU()
    
    if net_name == 'cycif_Net':
        net = CyCIF_VanillaNetwork()

    if net_name == 'cycif_ResNet':
        net = CyCIF_ResidualNetwork()

    return net 


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'cycif_Net', 'cycif_ResNet')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet_ELU':
        ae_net = CIFAR10_LeNet_ELU_Autoencoder()
    
    if net_name == 'cycif_Net':
        ae_net = CyCIF_VanillaAE()

    if net_name == 'cycif_ResNet':
        ae_net = CyCIF_ResNetAE()

    return ae_net
