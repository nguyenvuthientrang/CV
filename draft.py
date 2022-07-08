from dataloaders.get_data import get_dataset
from arguments import get_argparser
from torch.utils import data
import torch

opts = get_argparser().parse_args()

dataset_dict = get_dataset(opts)
train_loader = data.DataLoader(
    dataset_dict['train'], batch_size=opts.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
val_loader = data.DataLoader(
    dataset_dict['val'], batch_size=opts.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = data.DataLoader(
    dataset_dict['test'], batch_size=opts.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)

print("Dataset: %s, Train set: %d, Val set: %d, Test set: %d" %
        (opts.dataset, len(dataset_dict['train']), len(dataset_dict['val']), len(dataset_dict['test'])))

torch.save(train_loader, "train_loader.pkt")