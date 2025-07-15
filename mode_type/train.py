import torch
import numpy as np
from math import ceil
from os.path import join
import h5py
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from os import remove
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

def train(opt,model,
          train_set,
          device,
          optimizer,
          writer,
          criterion,
          epoch,
          scaler):

    if opt.dataset.lower() == 'aerialvl':
        import dataloader.aerialvl as dataset
    else:
        raise Exception('Unknown dataset')

    epoch_loss = 0
    startIter = 1 # keep track of batch iter across subsets for logging
    if opt.cacheRefreshRate > 0:
        subsetN = ceil(len(train_set) / opt.cacheRefreshRate)
        #TODO randomise the arange before splitting?
        subsetIdx = np.array_split(np.arange(len(train_set)), subsetN)
    else:
        subsetN = 1
        subsetIdx = [np.arange(len(train_set))]

    nBatches = (len(train_set) + opt.batchSize - 1) // opt.batchSize


    for subIter in range(subsetN):

        sub_train_set = Subset(dataset=train_set, indices=subsetIdx[subIter])

        training_data_loader = DataLoader(dataset=sub_train_set, num_workers=opt.threads,
                                          batch_size=opt.batchSize, shuffle=True,
                                          collate_fn=dataset.collate_fn, pin_memory=False)

        model.train()

        for iteration, (query, positives, negatives,
                        negCounts, indices) in enumerate(tqdm(training_data_loader, desc="Processing", unit="item", ncols=80), startIter):
            # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
            # where N = batchSize * (nQuery + nPos + nNeg)
            if query is None: continue # in case we get an empty batch

            B, C, H, W = query.shape
            nNeg = torch.sum(negCounts)
            input = torch.cat([query, positives, negatives.view(-1, C, H, W)])
            input = input.to(device)
            optimizer.zero_grad()
            loss = 0
            with autocast():
                aggregate_encoding = model(input)
                vladQ, vladP, vladN = torch.split(aggregate_encoding, [B, B, negatives.shape[1]*B])
                for i, negCount in enumerate(negCounts):
                    for n in range(negCount):
                        negIx = (torch.sum(negCounts[:i]) + n).item()
                        loss += criterion(vladQ[i:i+1], vladP[i:i+1], vladN[negIx:negIx+1])

                loss /= nNeg.float().to(device) # normalise by actual number of negatives
            # loss.backward()
            scaler.scale(loss).backward() 
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()  
            del input, aggregate_encoding, vladQ, vladP, vladN
            del query, positives, negatives

            batch_loss = loss.item()
            epoch_loss += batch_loss

        startIter += len(training_data_loader)
        del training_data_loader, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()

    avg_loss = epoch_loss / nBatches

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss),
          flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)