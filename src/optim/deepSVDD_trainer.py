from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
import scipy
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap
from yellowbrick.cluster import SilhouetteVisualizer
import os
from .early_stop import EarlyStopping

import logging
import time
import torch
import torch.optim as optim
import numpy as np
import pickle


class DeepSVDDTrainer(BaseTrainer):

    def __init__(self, xp_path, objective, R, c, nu: float, K: int, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)
        
        self.xp_path = xp_path

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu
        self.K = K

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get train data loader
        train_loader, _, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        es = EarlyStopping(patience=10)
        net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                
                ### NEW - get closest cluster center, take dist, sum/mean for loss
                if (self.c.dim() == 1): # naive deep_svdd
                    centers = self.c
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                else:
                    centers = torch.transpose(self.c,0,1)
                    dist = torch.zeros(outputs.shape[0], device=self.device)
                    for i in range(outputs.shape[0]):
                        # Sum dists from each data point to its corresponding cluster
                        # dist[i] = torch.sum((centers - outputs[i]) ** 2, dim=1).min()
                        ### Avoid gradient of min
                        cluster_idx = torch.sum((centers - outputs[i]) ** 2, dim=1).argmin()
                        dist[i] = torch.sum((centers[cluster_idx] - outputs[i]) ** 2, dim=0)
                ### 
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                n_batches += 1
            
            # ### DEBUG UMAPs and CENTERS (ADDED EPOCH NUMBER TO LATENT UMAP FUNCTION ###
            # train_loader, _, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
            
            # output_data = []
            # label_data = []
            # with torch.no_grad():
            #     for data in train_loader:
            #         # get the inputs of the batch
            #         inputs, labels, _ = data #labels are only for UMAP of hyperspheres
            #         inputs = inputs.to(self.device)
            #         outputs = net(inputs)
            #         output_data.append(outputs)
            #         label_data.append(labels)
            # kmeans_centers = np.load(os.path.join(self.xp_path,'centers.npy'))
            # output_data = torch.cat(output_data)
            # label_data = torch.cat(label_data).numpy()
            # self.latent_UMAP(epoch, output_data, label_data, kmeans_centers, pretrain_ae = False) ### USE pretrain_ae = False, no repeat
            # ############

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

            # Validation and Early Stopping
            _, val_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
            if val_loader.dataset is None:
                logger.info('Validation data not available...')
                pass
            else:
                net.eval()
                with torch.no_grad():
                    loss_val = 0.0
                    n_batches = 0
                    for data in val_loader:
                        inputs, _, _ = data
                        inputs = inputs.to(self.device)
                        outputs = net(inputs)
                        if (self.c.dim() == 1): # naive deep_svdd
                            centers = self.c
                            dist = torch.sum((outputs - self.c) ** 2, dim=1)
                        else:
                            centers = torch.transpose(self.c,0,1)
                            dist = torch.zeros(outputs.shape[0], device=self.device)
                            for i in range(outputs.shape[0]):
                                # Sum dists from each data point to its corresponding cluster
                                # dist[i] = torch.sum((centers - outputs[i]) ** 2, dim=1).min()
                                ### Avoid gradient of min
                                cluster_idx = torch.sum((centers - outputs[i]) ** 2, dim=1).argmin()
                                dist[i] = torch.sum((centers[cluster_idx] - outputs[i]) ** 2, dim=0)
                        if self.objective == 'soft-boundary':
                            scores = dist - self.R ** 2
                            loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                        else:
                            loss = torch.mean(dist)
                        
                        loss_val += loss.item()
                        n_batches += 1
                logger.info('  Validation Loss: {:.8f}'.format(loss_val / n_batches))
                if es.step( torch.tensor(loss_val / n_batches) ):
                    logger.info('  Stop early at epoch {}/{}'.format(epoch + 1, self.n_epochs))
                    break

        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        # # UMAP Plot (on training data)
        # # Get train data loader
        # train_loader, _, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        
        # output_data = []
        # label_data = []
        # with torch.no_grad():
        #     for data in train_loader:
        #         # get the inputs of the batch
        #         inputs, labels, _ = data #labels are only for UMAP of hyperspheres
        #         inputs = inputs.to(self.device)
        #         outputs = net(inputs)
        #         output_data.append(outputs)
        #         label_data.append(labels)
        # kmeans_centers = np.load(os.path.join(self.xp_path,'centers.npy'))
        # output_data = torch.cat(output_data)
        # label_data = torch.cat(label_data).numpy()
        # self.latent_UMAP(output_data, label_data, kmeans_centers, pretrain_ae = False) ### USE pretrain_ae = False, no repeat

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        _, _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        output_data = []
        label_data = []
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                output_data.append(outputs)
                label_data.append(labels)

                ### NEW
                if (self.c.dim() == 1): # naive deep_svdd
                    centers = self.c
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                else:
                    centers = torch.transpose(self.c,0,1)
                    dist = torch.zeros(outputs.shape[0], device=self.device)
                    for i in range(outputs.shape[0]):
                        # Sum dists from each data point to its corresponding cluster
                        # dist[i] = torch.sum((centers - outputs[i]) ** 2, dim=1).min()
                        ### Avoid gradient of min
                        cluster_idx = torch.sum((centers - outputs[i]) ** 2, dim=1).argmin()
                        dist[i] = torch.sum((centers[cluster_idx] - outputs[i]) ** 2, dim=0)
                ###
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        self.test_auc = roc_auc_score(labels, scores)
        logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))

        # #UMAP (same umap model fit in training) - use anomaly_data = True
        # # Plot with testing data
        # kmeans_centers = np.load(os.path.join(self.xp_path,'centers.npy'))
        # output_data = torch.cat(output_data)
        # label_data = torch.cat(label_data).numpy()
        # self.latent_UMAP(output_data, label_data, kmeans_centers, pretrain_ae = True, anomaly_data = True) ### USE pretrain_ae = False

        logger.info('Finished testing.')

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        logger = logging.getLogger()

        net.eval()
        if (self.K == 1): ### naive
            n_samples = 0 
            c = torch.zeros(net.rep_dim, device=self.device)
        else: ### multi-center code
            cen = torch.zeros(net.rep_dim, self.K, device=self.device)
            kmeans = KMeans(n_clusters=self.K,random_state=0)

        logger.info("Initializing {} clusters".format(self.K))

        output_data = []
        label_data = []
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, labels, _ = data #labels are only for UMAP of hyperspheres
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                output_data.append(outputs)
                label_data.append(labels)
                if (self.K == 1):
                    n_samples += outputs.shape[0] ### naive
                    c += torch.sum(outputs, dim=0) ### naive
            
            output_data = torch.cat(output_data)
            if (self.K == 1): ### naive
                c /= n_samples
                cen = c
            else: ### multi-center code
                kmeans = kmeans.fit(output_data)
                cluster_centers = torch.from_numpy(kmeans.cluster_centers_.T)
                cluster_centers = cluster_centers.type(torch.FloatTensor)
                cen = cluster_centers.to(self.device)
                dmat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(cen.detach().cpu().numpy().T))
                logger.info(f"Distances between cluster centers: \n{dmat}")

            # Generate silhouette plot
            ## self.silhouette_plot(output_data)

            # UMAP Plot
            label_data = torch.cat(label_data).numpy()
            if (self.K == 1): ### naive
                np.save(os.path.join(self.xp_path,'centers.npy'), cen.cpu().detach().numpy())
                self.latent_UMAP(output_data, label_data, cen.cpu().detach().numpy())
            else: ### multi-center code
                np.save(os.path.join(self.xp_path,'centers.npy'),kmeans.cluster_centers_)
                self.latent_UMAP(output_data, label_data, kmeans.cluster_centers_)
        
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        cen[(abs(cen) < eps) & (cen < 0)] = -eps
        cen[(abs(cen) < eps) & (cen > 0)] = eps

        return cen
    
    def latent_UMAP(self, latent_data, label_data, centers, pretrain_ae: bool = True, anomaly_data: bool = False, repeat: bool = False, epoch = -1):
        """UMAP of latent space and cluster hypersphere centers of Deep SVDD model
        
        :arg ANOMALY_DATA: set to True in testing, when data contains an anomaly class
        """

        # Add hypersphere centers to umap
        for i in range(self.K):
            label_data = np.append(label_data,-1)
            if (self.K == 1): # naive deep svdd
                latent_data = np.append(latent_data, np.expand_dims(centers,axis=0), axis=0)
            else: ### multi-center code
                latent_data = np.append(latent_data, np.expand_dims(centers[i],axis=0), axis=0)
        reducer = umap.UMAP()
        
        # save *trans* object for each K model. Need to fit only and save learned projection embedding
        if pretrain_ae: # embedding for pretrained AE
            trans = reducer.fit(latent_data)
            pickle.dump(trans, open(os.path.join(self.xp_path,f'{self.K}-cluster-umap-model.sav'), 'wb'))
            embedding = trans.embedding_
        else: # embedding for trained deep SVDD
            trans = pickle.load(open(os.path.join(self.xp_path,f'{self.K}-cluster-umap-model.sav'), 'rb'))
            trans = trans.transform(latent_data) # reduced representation
            embedding = trans
        # Plot
        if anomaly_data:
            plt.scatter(embedding[label_data==1, 0], embedding[label_data==1, 1], c='y',label='Anomaly', s = 10)
        plt.scatter(embedding[label_data==0, 0], embedding[label_data==0, 1], c='b',label='Normal', s = 10)
        plt.scatter(embedding[label_data == -1, 0], embedding[label_data == -1, 1], c='r', label='Center')
        plt.legend(loc="upper left")
        plt.gca().set_aspect('equal', 'datalim')
        plt.grid(False)
        ### if pretrain_ae and not anomaly_data: # embedding for pretrained AE
        if pretrain_ae and not anomaly_data and not repeat:
            plt.title('UMAP projection of the Deep SVDD Latent Space (Pre-trained AE)', fontsize=18)
            plt.savefig(os.path.join(self.xp_path,f'{self.K}-cluster-umap-scaled-pretrained-ae.png'),bbox_inches='tight')
        ### elif not pretrain_ae and not anomaly_data: # embedding for trained deep SVDD
        ### elif pretrain_ae and not anomaly_data and repeat:
        elif not pretrain_ae and not anomaly_data and not repeat:
            plt.title('UMAP projection of the Deep SVDD Latent Space (Trained Model)', fontsize=18)
            plt.savefig(os.path.join(self.xp_path,f'{self.K}-cluster-umap-scaled-trained-model-epoch-{epoch}.png'),bbox_inches='tight')
        ### elif not pretrain_ae and anomaly_data: # embedding for trained deep SVDD on test data
        elif pretrain_ae and anomaly_data:
            plt.title('UMAP projection of Deep SVDD Trained Model on Test Data', fontsize=18)
            plt.savefig(os.path.join(self.xp_path,f'{self.K}-cluster-umap-scaled-trained-model-test-data.png'),bbox_inches='tight')
        else:      
            pass # no condition for pretrained AE with test data
        plt.close()

    def silhouette_plot(self, latent_data):
        """Silhouette Plots and Scores to determine optimal K in KMeans""" 
        fig, ax = plt.subplots(2, 2, figsize=(15,8))
        for i in [2, 3, 4, 5]:
            '''
            Create KMeans instance for different number of clusters
            '''
            km = KMeans(n_clusters=i,random_state=0)
            q, mod = divmod(i, 2)
            '''
            Create SilhouetteVisualizer instance with KMeans instance
            Fit the visualizer
            '''
            visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
            visualizer.fit(latent_data)
        
        fig.suptitle('Silhouette Plots for 2, 3, 4, 5 Cluster Centers', fontsize=18)
        plt.savefig(os.path.join(self.xp_path,'Silhouette-Visualization.png'),bbox_inches='tight')
        plt.close('all')


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)