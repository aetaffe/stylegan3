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
    stats_path = '/media/alex/1TBSSD/research_gans/HPC/00023-stylegan2-FLIm-Images-no-phantom-512x512-gpus1-batch32-gamma10/stats.jsonl'
    stats = get_stats_dict(stats_path)
    loss_g_mean = []
    loss_d_mean = []
    k_imgs = []
    for item in stats['Loss/G/loss']:
        loss_g_mean.append(item['mean'])
    for item in stats['Loss/D/loss']:
        loss_d_mean.append(item['mean'])
    for item in stats['Progress/kimg']:
        k_imgs.append(item['mean'])
    plt.plot(k_imgs, loss_g_mean, label='Generator Loss')
    plt.plot(k_imgs, loss_d_mean, label='Discriminator Loss')
    plt.xlabel('Images Processed (x1000)')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs Images Processed (x1000)')
    plt.savefig('loss_vs_kimgs_gamma10.png')
    plt.show()




