import click
import dnnlib
import torch
from metrics import metric_main
import legacy
from typing import Optional

def get_training_set_kwargs(train_data_path: str) -> dict:
    dataset_class = 'training.dataset.ImageSegmentationDataset'
    dataset_kwargs = dnnlib.EasyDict(class_name=dataset_class, path=train_data_path, use_labels=False, max_size=None, xflip=False)
    return dataset_kwargs

def get_fid(G, D, train_data_path, score_threshold=10):
    training_set_kwargs = get_training_set_kwargs(train_data_path)
    return metric_main.calc_metric(metric='fid50k_full_threshold', G=G, D=D, dataset_kwargs=training_set_kwargs,
                                           device=torch.device('cuda'), score_threshold=score_threshold)

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--train-data', help='Path to training data for fid calculation', type=str, required=False)
def calculate_fid_threshold(
    network_pkl: str,
    train_data: Optional[str],
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        models = legacy.load_network_pkl(f)
        G = models['G_ema'].to(device=device)
        D = models['D'].to(device=device)
    thresholds = [20, 25, 30]
    for threshold in thresholds:
        results = get_fid(G, D, train_data, score_threshold=threshold)
        print(results)



if __name__ == '__main__':
    calculate_fid_threshold()