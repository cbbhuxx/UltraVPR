import torch
import json
import argparse
import logging
from datetime import datetime
import load_dataset
import utils
import torch.optim as optim
import torch.nn as nn
from mode_type import test, train, classes
from tensorboardX import SummaryWriter
from os.path import join
from os import makedirs
import warnings
from torch.cuda.amp import GradScaler, autocast

num_devices = torch.cuda.device_count()
parser = argparse.ArgumentParser(description='pytorch-locNet')
parser.add_argument('--upscaling', type=bool, default=False, help='upscaling for encoding')
parser.add_argument('--num_classes', type=list, default=[1, 2, 4, 8, 16, 32], help='Number of classes clusters. Default=10')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test'])
parser.add_argument('--batchSize', type=int, default=4,
                    help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
parser.add_argument('--dataset', type=str, default='aerialvl',
                    help='Dataset to use', choices=['aerialvl',])
parser.add_argument('--backbone', type=str, default='e2resnet50_c8', help='type of backbone to use',
                    choices=['e2resnet50_c8', 'e2resnet50_c16'])
parser.add_argument('--pooling', type=str, default='se2gem', help='type of pooling to use',
                    choices=['se2gem'])
parser.add_argument('--nEpochs', type=int, default=80, help='number of epochs to train for')
parser.add_argument('--cacheBatchSize', type=int, default=24, help='Batch size for caching and testing')
parser.add_argument('--cacheRefreshRate', type=int, default=12000,
                    help='How often to refresh cache, in number of queries. 0 for off')
parser.add_argument('--optim', type=str, default='SGD', help='optimizer to use', choices=['SGD', 'ADAM'])
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=13, help='Number of threads for each data loader to use')
parser.add_argument('--lr', type=float, default=0.0032, help='Learning Rate.')
parser.add_argument('--lrStep', type=list, default=20, help='Decay LR ever N steps.')
parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss. Default=0.1')
parser.add_argument('--num_clusters', type=int, default=32, help='Number of clusters. Default=64')
parser.add_argument('--savePath', type=str, default='checkpoints',
                    help='Path to save checkpoints to in logdir. Default=checkpoints/')
parser.add_argument('--runsPath', type=str, default='./runPath/', help='Path to save runs to.')
parser.add_argument('--dataPath', type=str, default='./dataPath/', help='Path for centroid data.')
parser.add_argument('--resume', type=str, default='', help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--cachePath', type=str, default="./cachePath/", help='Path to save cache to.')
parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping. 0 is off.')
parser.add_argument('--encoder_dim', type=int, default=256, help='Dimensions of Post-Backbone Network Characterization')



if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    opt = parser.parse_args()
    logging.info(opt)

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")
    device = torch.device("cuda" if cuda else "cpu")

    scaler = GradScaler()

    logging.info(f"Create a VPR model")
    mymodel = utils.set_model(opt)
    mymodel = mymodel.to(device)

    isParallel = False

    optimizer = optim.SGD(mymodel.parameters(), lr=opt.lr,
                          momentum=opt.momentum,
                          weight_decay=opt.weightDecay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)

    criterion = nn.TripletMarginLoss(margin=opt.margin ** 0.5,
                                     p=2, reduction='sum').to(device)

    if opt.mode.lower() == 'test':
        epoch = 1
        whole_test_set, whole_val_set = load_dataset.load_dataset(opt)
        recalls = test.test(opt,
                            mymodel,
                            whole_test_set,
                            whole_val_set,
                            device)

    elif opt.mode.lower() == 'train':
        train_set, whole_test_set, whole_val_set, classes_set = load_dataset.load_dataset(opt)
        writer = SummaryWriter(log_dir=join(opt.runsPath, datetime.now().strftime('%b%d_%H-%M-%S')+'_'+opt.backbone+'_'+opt.pooling))
        logdir = writer.file_writer.get_logdir()
        opt.savePath = join(logdir, opt.savePath)
        makedirs(opt.savePath)

        with open(join(opt.savePath, 'flags.json'), 'w') as f:
            f.write(json.dumps(
                {k:v for k,v in vars(opt).items()}
            ))
        not_improved = 0
        best_score = 0

        for epoch in range(1, opt.nEpochs + 1):
            if opt.optim.upper() == 'SGD':
                scheduler.step(epoch)
            if (epoch-1) % 1 == 0 :
                train_set, whole_test_set, whole_val_set, classes_set = load_dataset.load_dataset(opt, vpr_model=mymodel)
                if opt.upscaling:
                    classes_centers = classes.get_classes(opt,
                                mymodel,
                                classes_set,
                                device)
                    mymodel.upscale.init_params(classes_centers)
                
            train.train(opt,
                        mymodel,
                        train_set,
                        device,
                        optimizer,
                        writer,
                        criterion,
                        epoch,
                        scaler)

            if (epoch % 1) == 0:
                recalls = test.test(opt,
                                    mymodel,
                                    whole_test_set,
                                    whole_val_set,
                                    device)
                is_best = recalls[1] > best_score
                if is_best:
                    not_improved = 0
                    best_score = recalls[1]
                else:
                    not_improved += 1

                utils.save_checkpoint(opt,{
                    'epoch': epoch,
                    'state_dict': mymodel.state_dict(),
                    'recalls': recalls,
                    'best_score': best_score,
                    'optimizer' : optimizer.state_dict(),
                    'parallel' : isParallel,
                }, is_best)

                if opt.patience > 0 and not_improved > (opt.patience / 1):
                    print('Performance did not improve for', opt.patience, 'epochs. Stopping.')
                    break

        print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
        writer.close()
