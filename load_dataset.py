import logging
from torch.utils.data import DataLoader

def load_dataset(opt, vpr_model=None):
    
    if opt.dataset == 'aerialvl':
        import dataloader.aerialvl as dataset
    else:
        raise Exception('Unknown dataset')

    if opt.mode.lower() == "train":
        train_set = dataset.get_training_query_set(vpr_model=vpr_model)
        # print("train number:", len(train_set))
        whole_test_set = dataset.get_whole_test_set()      
        whole_val_set = dataset.get_whole_val_set()
        # print("test number:", len(whole_test_set))
        # print("val  number:", len(whole_val_set))
        classes_set = dataset.get_training_classes_set()
        # print("classes number:", len(classes_set))
        return train_set, whole_test_set,whole_val_set, classes_set

    elif opt.mode.lower() == "test":
        whole_test_set = dataset.get_whole_test_set()
        whole_val_set = dataset.get_whole_val_set()
        return whole_test_set, whole_val_set

    elif opt.mode.lower() == "clusters":
        whole_train_set = dataset.get_training_negativies_set()
        # logging.info(f"Total number of clustered datasets:{len(whole_train_set)}")
        return whole_train_set



