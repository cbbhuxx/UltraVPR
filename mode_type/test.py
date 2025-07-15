from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
import logging
import faiss

def test(opt,model,test_set, val_set, device):
    test_data_loader = DataLoader(dataset=test_set,
                                  num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False,
                                  pin_memory=False)
    val_data_loader = DataLoader(dataset=val_set,
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False,
                pin_memory=False)

    model.eval()
    if opt.pooling.lower() == 'se2gem':
        dim_ = opt.encoder_dim
    else:
        raise Exception('Unknown pooling')

    with torch.no_grad():
        pool_size = dim_
        dbFeat = np.empty((len(val_set), pool_size))
        qFeat = np.empty((len(test_set), pool_size))

        for iteration, (input, indices) in enumerate(tqdm(val_data_loader, desc="val_data:", unit="item", ncols=80), 1):
            input = input.to(device)
            image_encoding = model.backbone(input)
            aggregate_encoding = model.aggregator(image_encoding)
            dbFeat[indices.detach().numpy(), :] = aggregate_encoding.detach().cpu().numpy()
            del input, aggregate_encoding

        for iteration1, (input, indices) in enumerate(tqdm(test_data_loader, desc="test_data:", unit="item", ncols=80), 1):
            input = input.to(device)
            image_encoding = model.backbone(input)
            aggregate_encoding = model.aggregator(image_encoding)
            qFeat[indices.detach().numpy(), :] = aggregate_encoding.detach().cpu().numpy()
            del input, aggregate_encoding
    del test_data_loader, val_data_loader

    qFeat = qFeat.astype('float32')
    dbFeat = dbFeat.astype('float32')

    logging.debug(f"====> Building faiss index'")
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(dbFeat)

    print('====> Calculating recall @ N')
    n_values = [1,5,10]
    top_k = [1,5,10]

    _, predictions = faiss_index.search(qFeat, max(n_values))

    recalls = {}
    for h, j in enumerate(n_values):
        correct_at_n = 0
        for i in range(test_set.num):
            for k in range(j):
                if(abs(test_set.cc_index[i]-val_set.cc_index[predictions[i][k]]) <= 1):
                    correct_at_n += 1
                    break
        recall_at_n = correct_at_n / test_set.num
        recalls[j] = recall_at_n
        print("====>Recall@top-{}: {:.4f}".format(top_k[h], recall_at_n))

    return recalls