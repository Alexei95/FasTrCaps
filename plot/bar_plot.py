import matplotlib.pyplot as plt
import matplotlib

font = {
    'weight': 'normal',
    'size': 18}
linewidth = 3
width = 0.35

matplotlib.rc('font', **font)

DATA = {'accuracy': {'Fixed': 99.37,
                     'ExpDecay': 99.40,
                     'OCP': 99.38,
                     'WR': 99.44,
                     'AdaBatch': 99.41},
        'training_time': {'Fixed': 29,
                          'ExpDecay': 12,
                          'OCP': 24,
                          'WR': 11,
                          'AdaBatch': 8}}

LABELS = {}

SETTINGS = {'accuracy': {'y_label': 'Accuracy (%)',
                         'lightcolor': 'xkcd:azure',
                         'color': 'b',
                         'function': min,
                         'max_correction_factor': 1.0001,
                         'min_correction_factor': 0.9997,
                         'vertical_arrow': {'x_min': 3 - width / 2,
                                            'x_max': 3 - width / 2,
                                            'y_min': min(list(DATA['accuracy'].values())) * 0.99999,
                                            'y_max': max(list(DATA['accuracy'].values())) * 1.00001, }},
            'training_time': {'y_label': 'Epochs to \nmax accuracy',
                              'lightcolor': 'darkgreen',
                              'color': 'g',
                              'function': max,
                              'max_correction_factor': 1.05,
                              'min_correction_factor': 0.5,
                              'vertical_arrow': {'x_min': 4 + width / 2,
                                                 'x_max': 4 + width / 2,
                                                 'y_min': min(list(DATA['training_time'].values())) * 0.96,
                                                 'y_max': max(list(DATA['training_time'].values())) * 1.01, }}}

# DATA['accuracy'].update({'WarmAdaBatch': 99.52})
# DATA['training_time'].update({'WarmAdaBatch': 3})


fig, ax2 = plt.subplots(figsize=(12, 2.7), dpi=200)
ax1 = plt.twinx()

rects1 = ax1.bar([x - width/2 for x in range(len(DATA['accuracy']))], list(
    DATA['accuracy'].values()), width, label='Accuracy', edgecolor='k', color=SETTINGS['accuracy']['color'])
rects2 = ax2.bar([x + width/2 for x in range(len(DATA['training_time']))], list(
    DATA['training_time'].values()), width, label='Training time', edgecolor='k', color=SETTINGS['training_time']['color'])

plt.xticks(range(len(DATA['accuracy'])), labels=list(DATA['accuracy'].keys()))

SETTINGS['accuracy']['axes'] = ax1
SETTINGS['training_time']['axes'] = ax2

for k, elem in DATA.items():

    ax = SETTINGS[k]['axes']

    # ax.legend(['OCP'])

    max_y = max(list(elem.values())) * SETTINGS[k]['max_correction_factor']
    min_y = min(list(elem.values())) * SETTINGS[k]['min_correction_factor']

    ax.axhline(SETTINGS[k]['function'](list(elem.values())),
               linestyle='dashed', linewidth=linewidth, color=SETTINGS[k]['lightcolor'])
    # plt.axvline(SETTINGS[k]['vertical_arrow']['index'],
    #             ymin=SETTINGS[k]['vertical_arrow']['span_min'],
    #             ymax=SETTINGS[k]['vertical_arrow']['span_max'],
    #             linestyle='dashed',
    #             color='r',
    #             dash_capstyle='projecting',
    #             linewidth=linewidth)
    # plt.arrow(, 0.2 * 4.7, 0.5 * 8, 0.5 * 4.7)
    ax.annotate("",
                xy=(SETTINGS[k]['vertical_arrow']['x_min'],
                    SETTINGS[k]['vertical_arrow']['y_min']),
                xytext=(SETTINGS[k]['vertical_arrow']['x_max'],
                        SETTINGS[k]['vertical_arrow']['y_max']),
                arrowprops=dict(arrowstyle="<->",
                                color='r', linewidth=linewidth))

    ax.set_ylim([min_y, max_y])

    ax.set_ylabel(SETTINGS[k]['y_label'], labelpad=14, fontdict={
        'fontsize': font['size'], 'fontweight': font['weight'], 'color': SETTINGS[k]['color']})
    ax.tick_params(axis='y', colors=SETTINGS[k]['color'])


props = dict(boxstyle='round', facecolor='white', alpha=0.1)
plt.text(1.25, 0.89, 'WarmRestarts shows the\nbest improvement w.r.t\nthe baseline for accuracy', transform=ax.transAxes,
         fontsize=font['size'] - 2,
         verticalalignment='top', bbox=props)
props = dict(boxstyle='round', facecolor='white', alpha=0.1)
# 0.71, 0.89
plt.text(1.25, 0.39, 'AdaBatch shows the best\nimprovement w.r.t the\nbaseline for training time', transform=ax.transAxes, fontsize=font['size'] - 2,
         verticalalignment='top', bbox=props)

# plt.tight_layout()

plt.subplots_adjust(left=0.125, right=0.6)

plt.savefig('bar.pdf', bbox_inches="tight", pad_inches=0.2)

plt.show()
