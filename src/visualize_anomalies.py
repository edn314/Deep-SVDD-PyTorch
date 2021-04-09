import click
import torch
import logging
import random
import numpy as np

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from utils.visualization.plot_images_labels import plot_images_labels
from deepSVDD import DeepSVDD
from datasets.main import load_dataset

################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['mnist', 'cifar10', 'cycif']))
@click.argument('net_name', type=click.Choice(['mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'cycif_Net']))
@click.argument('xp_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--load_ae_model', type=click.Path(exists=True), default=None,
              help='Pretrained AE model file path (default: None).')
@click.option('--objective', type=click.Choice(['one-class', 'soft-boundary']), default='one-class',
              help='Specify Deep SVDD objective ("one-class" or "soft-boundary").')
@click.option('--nu', type=float, default=0.1, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
@click.option('--number_clusters', type=int, default=1, help='Deep SVDD number of cluster centers.')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for Deep SVDD network training.')
@click.option('--lr', type=float, default=0.001,
              help='Initial learning rate for Deep SVDD network training. Default=0.001')
@click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.')
@click.option('--pretrain', type=bool, default=True,
              help='Pretrain neural network parameters via autoencoder.')
@click.option('--ae_optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for autoencoder pretraining.')
@click.option('--ae_lr', type=float, default=0.001,
              help='Initial learning rate for autoencoder pretraining. Default=0.001')
@click.option('--ae_n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
@click.option('--ae_lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--ae_batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
@click.option('--ae_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
def main(dataset_name, net_name, xp_path, data_path, load_config, load_model, load_ae_model, objective, nu, number_clusters, device, seed,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, pretrain, ae_optimizer_name, ae_lr,
         ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay, n_jobs_dataloader, normal_class):
    """
    Deep SVDD, a fully deep method for anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)
    
    # Print configuration
    logger.info('Deep SVDD objective: %s' % cfg.settings['objective'])
    logger.info('Nu-paramerter: %.2f' % cfg.settings['nu'])
    logger.info('Number of hyperphere centers: %d' % cfg.settings['number_clusters'])
    
    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    dataset = load_dataset(dataset_name, data_path, normal_class)

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(cfg.settings['objective'], cfg.settings['nu'], cfg.settings['number_clusters'])
    deep_SVDD.set_network(net_name)
    # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    if load_model:
        deep_SVDD.load_model(model_path=load_model, load_ae=False)
        logger.info('Loading model from %s.' % load_model)

    # import pdb; pdb.set_trace()
    # Test model
    deep_SVDD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    import pdb; pdb.set_trace()

    # Plot most anomalous and most normal (within-class) test samples
    indices, labels, scores = zip(*deep_SVDD.results['test_scores'])
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    idx_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # sorted from lowest to highest anomaly score

    # NEW # 
    # idx_sorted_normal = indices[labels == 0][np.argsort(scores[labels == 0])]  # normal images sorted from lowest to highest anomaly score
    # idx_sorted_outlier = indices[labels == 1][np.argsort(scores[labels == 1])]  # anomaly images sorted from lowest to highest anomaly score

    # Lowest to highest uncertainty scores
    idx_sorted_all = indices[np.argsort(scores)]
    labels_sorted_all = labels[np.argsort(scores)]
    scores_sorted_all = np.sort(scores)

    for i in range(32):
        idx = idx_sorted_all[i]   
        X = dataset.test_set[idx][0].unsqueeze(1)
        plot_images_labels(X, label = labels_sorted_all[i], export_img=xp_path + '/simple_img_'+str(i), title='Simplest Example: Score = {:4.2f}'.format(scores_sorted_all[i]), padding=2)

    # Highest to lowest uncertainty scores
    idx_sorted_all = np.flip(idx_sorted_all)
    labels_sorted_all = np.flip(labels_sorted_all)
    scores_sorted_all = np.flip(scores_sorted_all)

    for i in range(32):
        idx = idx_sorted_all[i]
        X = dataset.test_set[idx][0].unsqueeze(1)
        plot_images_labels(X, label = labels_sorted_all[i], export_img=xp_path + '/difficult_img_'+str(i), title='Difficult Example: Score = {:4.2f}'.format(scores_sorted_all[i]), padding=2)

    import pdb; pdb.set_trace()

    # X_n = [dataset.test_set[i][0] for i in idx_sorted_normal[-8:]]
    # X_n = torch.cat(X_n).unsqueeze(1)
    # X_o = [dataset.test_set[i][0] for i in idx_sorted_outlier[-8:]]
    # X_o = torch.cat(X_o).unsqueeze(1)

    # # import pdb; pdb.set_trace()
    # plot_images_labels(X_n, label = 0, export_img=xp_path + '/normals', title='Hardest normal examples', padding=2)
    # # import pdb; pdb.set_trace()
    # plot_images_labels(X_o, label = 1, export_img=xp_path + '/outliers', title='Hardest outlier examples', padding=2)
    #-#

    # From clean images, extract the ones model predicts as normal with highest confidence
    X_normals = [dataset.test_set[i][0] for i in idx_sorted[:64]]
    X_normals = torch.cat(X_normals).unsqueeze(1)

    # From clean images, extract the ones model predicts as normal with lowest confidence
    X_outliers = [dataset.test_set[i][0] for i in idx_sorted[-64:]]
    X_outliers = torch.cat(X_outliers).unsqueeze(1)

    plot_images_grid(X_normals, export_img=xp_path + '/normals_64', title='Most normal examples', padding=2)
    plot_images_grid(X_outliers, export_img=xp_path + '/outliers_64', title='Most anomalous examples', padding=2)

if __name__ == '__main__':
    main()
    
    #python visualize_anomalies.py cycif cycif_Net ../log/cycif-run-test1 /n/pfister_lab2/Lab/enovikov/data/Artifact-CyCIF-Data-2021/Sardana-Annotations/Edward/train/data/ --load_config /n/pfister_lab2/Lab/enovikov/unsup-ano-detection/anomaly-project/Deep-SVDD-PyTorch/log/cycif-run7-multi-center-k-2-update-with-pretrain/config.json --load_model /n/pfister_lab2/Lab/enovikov/unsup-ano-detection/anomaly-project/Deep-SVDD-PyTorch/log/cycif-run7-multi-center-k-2-update-with-pretrain/model.tar --objective one-class --lr 0.0001 --n_epochs 300 --lr_milestone 50 --batch_size 1 --weight_decay 0.5e-6 --pretrain False
    #python visualize_anomalies.py cycif cycif_Net ../log/cycif-run-test1 /n/pfister_lab2/Lab/enovikov/data/Artifact-CyCIF-Data-2021/Sardana-Annotations/Edward/train/data/ --load_config /n/pfister_lab2/Lab/enovikov/unsup-ano-detection/anomaly-project/Deep-SVDD-PyTorch/log/cycif-run8-multi-center-k-4-update-with-pretrain/config.json --load_model /n/pfister_lab2/Lab/enovikov/unsup-ano-detection/anomaly-project/Deep-SVDD-PyTorch/log/cycif-run8-multi-center-k-4-update-with-pretrain/model.tar --objective one-class --lr 0.0001 --n_epochs 300 --lr_milestone 50 --batch_size 1 --weight_decay 0.5e-6 --pretrain False

