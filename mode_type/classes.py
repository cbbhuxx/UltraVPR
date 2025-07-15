from torch.utils.data import DataLoader
from os.path import join, exists
from os import makedirs, remove
import h5py
import torch
import numpy as np
import faiss
from tqdm import tqdm

def get_classes(opt, model, cluster_set, device ):
    classes_data_loader = DataLoader(dataset=cluster_set,
                                     num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False,
                                     pin_memory=False)
    classescache = join(opt.dataPath, 'centers',
                        opt.dataset + '_' + str(opt.num_classes) + '_' + opt.pooling + '_desc_cen.hdf5')
    if opt.pooling.lower() == 'se2gem':
        dim_ = opt.encoder_dim

    with h5py.File(classescache, mode='w') as h5:
        with torch.no_grad():
            model.eval()
            dbFeat = h5.create_dataset("Descriptors",
                                       [len(cluster_set), dim_],
                                       dtype=np.float32)
            for iteration, (input, indices) in enumerate(tqdm(classes_data_loader, desc="classesing", unit="item", ncols=100), 1):
                input = input.to(device)
                image_encoding = model.backbone(input)
                aggregate_encoding = model.aggregator(image_encoding)
                dbFeat[indices.detach().numpy(), :] = aggregate_encoding.detach().cpu().numpy()
                del input, image_encoding, aggregate_encoding

        niter = 100
        for i,classes in enumerate(opt.num_classes):
            kmeans = faiss.Kmeans(dim_, classes, niter=niter, verbose=False)
            kmeans.train(dbFeat[...])
            if i == 0:
                center = kmeans.centroids
                continue
            center = np.vstack((center, kmeans.centroids))
        print('====> Classes_clustering Done!')

        del h5['Descriptors']
        h5.close()
        return center