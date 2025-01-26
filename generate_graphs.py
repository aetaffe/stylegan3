import json
import matplotlib.pyplot as plt


def get_stats_dict(stats_path):
    stats_dict = {}
    with open(stats_path) as f:
        lines = f.readlines()
        for line in lines:
            stat_line = json.loads(line)
            for key, value in stat_line.items():
                data = stats_dict.get(key, [])
                data.append(value)
                stats_dict[key] = data
    return stats_dict


if __name__ == '__main__':
    stats_path = '/home/alex/research/SyntheticData/stylegan3/training-runs/00010-stylegan2-FLIm-Images-no-phantom-256x256-gpus1-batch32-gamma1.5/stats.jsonl'
    stats = get_stats_dict(stats_path)
    loss_g_mean = []
    loss_d_mean = []
    k_imgs = []
    for item in stats['Loss/G/loss']:
        loss_g_mean.append(item['mean'])
    for item in stats['Loss/D/loss']:
        loss_d_mean.append(item['mean'])
    for item in stats['Progress/kimg']:
        k_imgs.append(item['mean'] + 900)
    plt.plot(k_imgs, loss_g_mean, label='Generator Loss')
    plt.plot(k_imgs, loss_d_mean, label='Discriminator Loss')
    plt.xlabel('Images Processed (x1000)')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs Images Processed (x1000)')
    plt.savefig('loss_vs_kimgs_gamma.8192.png')
    plt.show()




