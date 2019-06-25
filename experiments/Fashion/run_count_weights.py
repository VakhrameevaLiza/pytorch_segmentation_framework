from get_config import get_config
config=get_config('')

from datasets.FashionDataset import FashionDataset


from utils.count_weights import count_weigths

train_dataset = FashionDataset(config.dataset_path, 'training', h=config.h, w=config.w, mean=config.mean,
                               train_pct=1.0)
count_weigths(train_dataset, config.experiment_dir)