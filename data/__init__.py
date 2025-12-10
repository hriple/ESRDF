'''create dataset and dataloader'''
import logging
from re import split
import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'val':
        # return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        return torch.utils.data.DataLoader(dataset, 
                                           batch_size=dataset_opt['batch_size'],
                                           shuffle=dataset_opt['use_shuffle'],
                                           num_workers=dataset_opt['num_workers'],
                                           pin_memory=True)
    else:
        raise NotImplementedError('Dataloader [{:s}] is not found.'.format(phase))

# shuffle 数据可能是管用的，但是 可能存在问题是 create_dataset 就只有那 data_len 张 数据
def create_dataset(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    # print("dataset_opt['data_len']",dataset_opt['data_len'])
    from data.LRHR_dataset import LRHRDataset as D
    dataset = D(dataroot=dataset_opt['dataroot'],
                datatype=dataset_opt['datatype'],
                l_resolution=dataset_opt['l_resolution'],
                r_resolution=dataset_opt['r_resolution'],
                # train or valid
                split=phase,
                data_len=dataset_opt['data_len'],
                # mode == 'LRHR' 就是 需要LR
                need_LR=(mode == 'LRHR')
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
