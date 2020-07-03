from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

with open(Path('tmp.txt'), 'r') as f:
    lines_read = f.readlines()

lines = list()
for line in lines_read:
    lines.append(line.split())

labels = list()
naive_time = list()
kapra_time = list()

for index, line in enumerate(lines):
    if index % 2 == 0: # naive
        labels.append(line[1])
        naive_time.append(float(line[2]))
    else: # kapra
        kapra_time.append(float(line[2]))


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, naive_time, width, label='Naive')
rects2 = ax.bar(x + width/2, kapra_time, width, label='Kapra')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Time (s)')
ax.set_xlabel('Number of instances')
ax.set_title('Time efficiency')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

#plt.show()
plt.savefig('stat.png')

