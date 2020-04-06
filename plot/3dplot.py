import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

font = {
    'weight': 'normal',
    'size': 18}

matplotlib.rc('font', **font)

# Name, accuracy (%), training time (avg epoch s), #parameters, marker
# 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
DATA = (('WarmAdaBatch', 99.454, 3 * 800, 8227248/1000000, 'o', (matplotlib.colors.to_rgba('tab:blue'), )),
        ('WarmRestarts', 99.44, 11 * 300, 8227248/1000000,
         'v', (matplotlib.colors.to_rgba('tab:pink'), )),
        ('AdaBatch', 99.41, 8 * 754, 8227248/1000000,
         'X', (matplotlib.colors.to_rgba('tab:green'), )),
        ('ExpDecay', 99.404, 12 * 300, 8227248/1000000,
         'D', (matplotlib.colors.to_rgba('tab:red'), )),
        ('OneCyclePolicy', 99.384, 24 * 300, 8227248/1000000,
         's', (matplotlib.colors.to_rgba('tab:purple'), )),
        ('BaselineCapsNet', 99.366, 29 * 300, 8227248/1000000,
         'p', (matplotlib.colors.to_rgba('darkgreen'), )),
        ('WeightSharing', 99.1, 27 * 300, 6719920/1000000,
         'P', (matplotlib.colors.to_rgba('tab:brown'), )),
        ('WAB+WS',
         99.38, 18 * 400, 6719920/1000000, '*', (matplotlib.colors.to_rgba('tab:orange'), )),
        )
PROJECTIONS = True
min_x = min(elem[1] for elem in DATA)
min_x = min_x - 0.001 * min_x
max_x = max(elem[1] for elem in DATA)
max_x = max_x + 0.0005 * max_x
min_y = min(elem[3] for elem in DATA)
min_y = min_y - 0.001 * min_y
max_y = max(elem[2] for elem in DATA)
max_y = max_y + 0.15 * max_y
min_z = min(elem[3] for elem in DATA)
min_z = min_z - 0.1 * min_z
max_z = max(elem[3] for elem in DATA)
max_z = max_z + 0.001 * max_z

fig = plt.figure(figsize=(15, 7), dpi=100)
ax = fig.add_subplot(2, 2, 1, projection='3d')
for elem in DATA:
    ax.scatter(*elem[1:4], label=elem[0], marker=elem[4], s=150, c=elem[5])
    # ax.text(*elem[1:], elem[0], zdir='x')  # puts the text along the point
    # for projections
    if PROJECTIONS:
        reduced_alpha = [list(elem[5][0])]
        reduced_alpha[0][3] = reduced_alpha[0][3] - 0.5
        ax.scatter(elem[1], elem[2], min_z,
                   marker=elem[4], s=40, c=reduced_alpha)
        ax.scatter(min_x, elem[2], elem[3],
                   marker=elem[4], s=40, c=reduced_alpha)
        ax.scatter(elem[1], max_y, elem[3],
                   marker=elem[4], s=40, c=reduced_alpha)
ax.set_position([-0.03, 0.37, 0.412, 0.63])  # to move the plot itself
# plt.legend(bbox_to_anchor=(1.05, 0.8),
#            prop={'size': font['size'] - 5})
# plt.legend(loc='center left', )  # for the legend
# plt.title('Comparison of different optimizations', fontdict={
#          'fontsize': font['size'], 'fontweight': font['weight']})
plt.xlabel('Accuracy [%]', labelpad=14, fontdict={
           'fontsize': font['size'], 'fontweight': font['weight']})
plt.ylabel('Training time (s)', labelpad=14, fontdict={
           'fontsize': font['size'], 'fontweight': font['weight']})
ax.set_zlabel('#parameters (millions)', labelpad=8, fontdict={
              'fontsize': font['size'], 'fontweight': font['weight']})
ax.set_xlim([min_x, max_x])
ax.set_ylim([min_y, max_y])
ax.set_zlim([min_z, max_z])
plt.title('           (a)', fontdict={'fontweight': 'bold'})


# Name, accuracy (%), training time (avg epoch s), #parameters, marker
# 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
DATA = (('WarmAdaBatch', 91.47, 27 * 310, 8227248/1000000, 'o', (matplotlib.colors.to_rgba('tab:blue'), )),
        ('WarmRestarts', 91.35, 17 * 290, 8227248/1000000,
         'v', (matplotlib.colors.to_rgba('tab:pink'), )),
        ('AdaBatch', 91.27, 27 * 350, 8227248/1000000,
         'X', (matplotlib.colors.to_rgba('tab:green'), )),
        ('ExpDecay', 91.65, 29 * 280, 8227248/1000000,
         'D', (matplotlib.colors.to_rgba('tab:red'), )),
        ('OneCyclePolicy', 91.73, 27 * 300, 8227248/1000000,
         's', (matplotlib.colors.to_rgba('tab:purple'), )),
        ('BaselineCapsNet', 90.99, 17 * 290, 8227248/1000000,
         'p', (matplotlib.colors.to_rgba('darkgreen'), )),
        ('WeightSharing', 90.47, 17 * 300, 6719920/1000000,
         'P', (matplotlib.colors.to_rgba('tab:brown'), )),
        ('WAB+WS',
         90.61, 20 * 310, 6719920/1000000, '*', (matplotlib.colors.to_rgba('tab:orange'), )),
        )
PROJECTIONS = True
min_x = min(elem[1] for elem in DATA)
min_x = min_x - 0.001 * min_x
max_x = max(elem[1] for elem in DATA)
max_x = max_x + 0.0005 * max_x
min_y = min(elem[3] for elem in DATA)
min_y = min_y - 0.001 * min_y
max_y = max(elem[2] for elem in DATA)
max_y = max_y + 0.15 * max_y
min_z = min(elem[3] for elem in DATA)
min_z = min_z - 0.1 * min_z
max_z = max(elem[3] for elem in DATA)
max_z = max_z + 0.001 * max_z

ax = fig.add_subplot(2, 2, 2, projection='3d')
for elem in DATA:
    ax.scatter(*elem[1:4], label=elem[0], marker=elem[4], s=150, c=elem[5])
    # ax.text(*elem[1:], elem[0], zdir='x')  # puts the text along the point
    # for projections
    if PROJECTIONS:
        reduced_alpha = [list(elem[5][0])]
        reduced_alpha[0][3] = reduced_alpha[0][3] - 0.5
        ax.scatter(elem[1], elem[2], min_z,
                   marker=elem[4], s=40, c=reduced_alpha)
        ax.scatter(min_x, elem[2], elem[3],
                   marker=elem[4], s=40, c=reduced_alpha)
        ax.scatter(elem[1], max_y, elem[3],
                   marker=elem[4], s=40, c=reduced_alpha)
ax.set_position([0.555, 0.37, 0.412, 0.63])  # to move the plot itself
plt.legend(bbox_to_anchor=(0.07, 0.6),
           prop={'size': font['size'] - 3})
# plt.legend(loc='center left', )  # for the legend
# plt.title('Comparison of different optimizations', fontdict={
#          'fontsize': font['size'], 'fontweight': font['weight']})
plt.xlabel('Accuracy [%]', labelpad=17, fontdict={
           'fontsize': font['size'], 'fontweight': font['weight']})
plt.ylabel('Training time (s)', labelpad=14, fontdict={
           'fontsize': font['size'], 'fontweight': font['weight']})
plt.xticks(rotation=15)
ax.set_zlabel('#parameters (millions)', labelpad=8, fontdict={
              'fontsize': font['size'], 'fontweight': font['weight']})
ax.set_xlim([min_x, max_x])
ax.set_ylim([min_y, max_y])
ax.set_zlim([min_z, max_z])
plt.title('           (b)', fontdict={'fontweight': 'bold'})

DATA = (('MNIST/warmada_custom', '-o', (matplotlib.colors.to_rgba('tab:blue'), )),
        ('MNIST/warm',
         '-v', (matplotlib.colors.to_rgba('tab:pink'), )),
        ('MNIST/adabatch',
         '-X', (matplotlib.colors.to_rgba('tab:green'), )),
        ('MNIST/exp',
         '-D', (matplotlib.colors.to_rgba('tab:red'), )),
        ('MNIST/onecycle',
         '-s', (matplotlib.colors.to_rgba('tab:purple'), )),
        ('MNIST/baseline',
         '-p', (matplotlib.colors.to_rgba('darkgreen'), )),
        ('MNIST/ws', '-P', (matplotlib.colors.to_rgba('tab:brown'), )),
        ('MNIST/wab_ws',
         '-*', (matplotlib.colors.to_rgba('tab:orange'), )),
        )
ax = fig.add_subplot(2, 2, 3)
filename = 'test_accuracy.txt'
for elem in DATA:
    accuracies = []
    acc_av = []
    for i in range(1, 6):
        with open(elem[0] + '/' + str(i) + '/' + filename, 'r') as f:
            accuracies.append([float(x) for x in f.readlines()])
    for i in range(len(accuracies[0])):
        acc_av.append(sum(x[i] for x in accuracies) / len(accuracies))
    ax.plot(range(1, len(acc_av)+1), acc_av,
            elem[1], color=elem[2][0])
ax.set_position([0.07, 0.1, 0.392, 0.2])
plt.xlabel('Epochs', fontdict={
           'fontsize': font['size'], 'fontweight': font['weight']})
plt.ylabel('Accuracy [%]', fontdict={
           'fontsize': font['size'], 'fontweight': font['weight']})
plt.yticks([96, 97.5, 99])
ax.grid()
plt.title('                  (c)', fontdict={'fontweight': 'bold'})


DATA = (('FashionMNIST/fashionmnist_wab_0', '-o', (matplotlib.colors.to_rgba('tab:blue'), )),
        ('FashionMNIST/fashionmnist_warm_restarts_0',
         '-v', (matplotlib.colors.to_rgba('tab:pink'), )),
        ('FashionMNIST/fashionmnist_adabatch_0',
         '-X', (matplotlib.colors.to_rgba('tab:green'), )),
        ('FashionMNIST/fashionmnist_exp_decay_0',
         '-D', (matplotlib.colors.to_rgba('tab:red'), )),
        ('FashionMNIST/fashionmnist_one_cycle_policy_0',
         '-s', (matplotlib.colors.to_rgba('tab:purple'), )),
        ('FashionMNIST/fashionmnist_baseline_0',
         '-p', (matplotlib.colors.to_rgba('darkgreen'), )),
        ('FashionMNIST/fashionmnist_weight_sharing_0', '-P',
         (matplotlib.colors.to_rgba('tab:brown'), )),
        ('FashionMNIST/fashionmnist_wab_weight_sharing_0',
         '-*', (matplotlib.colors.to_rgba('tab:orange'), )),
        )
ax = fig.add_subplot(2, 2, 4)
filename = 'test_accuracy.txt'
for elem in DATA:
    accuracies = []
    with open(elem[0] + '/' + filename, 'r') as f:
        accuracies = [float(x) for x in f.readlines()]
    ax.plot(range(1, len(accuracies)+1), accuracies,
            elem[1], color=elem[2][0])
ax.set_position([0.55, 0.1, 0.412, 0.2])
ax.grid()
plt.yticks([80, 86, 92])
plt.xlabel('Epochs', fontdict={
           'fontsize': font['size'], 'fontweight': font['weight']})
plt.ylabel('Accuracy [%]', fontdict={
           'fontsize': font['size'], 'fontweight': font['weight']})
plt.title('                (d)', fontdict={'fontweight': 'bold'})

# plt.tight_layout()

#plt.savefig("ciao.pdf", bbox_inches="tight", pad_inches=0.2)
plt.savefig("final_comparison_with_accuracies.pdf")

plt.show()
