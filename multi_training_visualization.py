import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os import listdir
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing import event_accumulator

# Helpful links:
# https://stackoverflow.com/questions/41074688/how-do-you-read-tensorboard-files-programmatically

baseline_root_path = '/home/s6pereir/multirun_results/general_stkl_v2_results/August_2019/'
scenario = "particle_env_tag"
baseline_runs = [os.path.join(baseline_root_path, f) for f in listdir(baseline_root_path) if scenario in f]

print("baseline runs")
print(baseline_runs)

# get tensorboard events
baseline_tb_output_files = []
for f in baseline_runs:
    sub_dir = f+'/tb/'
    tb_outs = [os.path.join(sub_dir, r) for r in listdir(sub_dir)]
    baseline_tb_output_files.append(tb_outs)

print("baseline outputs")
print(baseline_tb_output_files)

event_accumulators = []
for tbo in baseline_tb_output_files[:5]:
    ea = event_accumulator.EventAccumulator(tbo[0],
        size_guidance={ # see below regarding this argument
        event_accumulator.COMPRESSED_HISTOGRAMS: 500,
        event_accumulator.IMAGES: 4,
        event_accumulator.AUDIO: 4,
        event_accumulator.SCALARS: 0,
        event_accumulator.HISTOGRAMS: 1,
    })
    ea.Reload()
    event_accumulators.append(ea)

print("Total event accumulators: ", len(event_accumulators))

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

plt.figure(figsize=[6, 6])
plt.ylabel('Values')
plt.xlabel('Steps')

for ea in event_accumulators:
    w_times, step_nums, vals = zip(*ea.Scalars('gym_highway/tb/ro/return_history'))
    vals_df = pd.DataFrame(np.array(vals))
    smooth_val = smooth(vals, 0.6)
    
    # plt.plot(vals_df[0], 'lightblue', smooth_val, 'b')
    plt.plot(smooth_val)

plt.savefig("test_general_stkl_v2_smooth0.6.svg")
#print(step_nums[:20])
# print("Total steps: ", len(step_nums))

exit(0)

# result_dir = '/home/s6pereir/gym_highway_results/August_2019/maddpg_assigned_ntw_stkl_general_14_17-17_iq1p/'
result_dir = '/home/s6pereir/multirun_results/baseline_results/August_2019/'
# ea = event_accumulator.EventAccumulator(result_dir+'tb/events.out.tfevents.1565817481.crowley8',
ea = event_accumulator.EventAccumulator(result_dir,
    size_guidance={ # see below regarding this argument
    event_accumulator.COMPRESSED_HISTOGRAMS: 500,
    event_accumulator.IMAGES: 4,
    event_accumulator.AUDIO: 4,
    event_accumulator.SCALARS: 0,
    event_accumulator.HISTOGRAMS: 1,
})
ea.Reload()

tags = ea.Tags()
print(tags)

#exit(0)

w_times, step_nums, vals = zip(*ea.Scalars('gym_highway/tb/ro/return_history'))

#print(step_nums[:20])
print(len(step_nums))

# exit(0)

plt.figure(figsize=[6, 6])
plt.plot(vals)
plt.ylabel('Values')
plt.xlabel('Steps')
# plt.gca().set_position([0, 0, 1, 1]) # this removes the axis values and labels
plt.savefig("test3.svg")


vals_df = pd.DataFrame(np.array(vals))

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


smooth_val = smooth(vals, 0.9)
smooth_val


plt.figure(figsize=[6, 6])
plt.plot(vals_df[0], 'lightblue', smooth_val, 'b')
plt.ylabel('Values')
plt.xlabel('Steps')
# plt.savefig("smooth_0.9.svg")



# ea.Scalars('gym_highway/tb/ro/return_history').Keys()
ea.scalars.Keys()


all_scalar_events = ea.scalars.Items('gym_highway/tb/ro/return_history')


all_steps_per_key = [[scalar_events.step, scalar_events.value, scalar_events.wall_time] for scalar_events in all_scalar_events]
all_steps_per_key


# function to plot data points
def plotData(data, axisLabels, title):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel(axisLabels[0], fontsize = 15)
    ax.set_ylabel(axisLabels[1], fontsize = 15)
    ax.set_title(title, fontsize = 20)
    targets = [0,1,2,3,4]
    colors = ['r', 'g', 'b','c','y']
    for target, color in zip(targets,colors):
        indicesToKeep = data['class'] == target
        ax.scatter(data.loc[indicesToKeep, axisLabels[0]]
                   , data.loc[indicesToKeep, axisLabels[1]]
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()


# function to plot images as points
def imscatter(x, y, image, axisLabels, title, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    x, y = np.atleast_1d(x, y)
    artists = []
    counter = 0
    for x0, y0 in zip(x, y):
        im = OffsetImage(image[counter], zoom=zoom, cmap=plt.cm.gray_r)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
        counter += 1
    ax.update_datalim(np.column_stack([x, y]))
    ax.set_xlabel(axisLabels[0], fontsize = 15)
    ax.set_ylabel(axisLabels[1], fontsize = 15)
    ax.set_title(title, fontsize = 20)
    ax.autoscale()
    return artists


# compute misclassificatoins ITERATIONS number of times to get an average
for i in range(ITERATIONS):
    errors = []
    for num_comp in num_components:
        # perform dimensionality reduction
        dimReducedDf = dimReduction(samples_std, num_comp)
        # split the data into 70% training and 30% test sets 
        X_train, X_test, y_train, y_test = train_test_split(dimReducedDf, dataB_gnd, test_size=0.3)
        # train using training sets, and predict labels of test set
        y_pred = gnb.fit(X_train.values, y_train).predict(X_test.values)
        # count the number of misclassified values
        errors.append((y_test.ravel() != y_pred).sum())
    
    # keep track of total number of misclassifications
    avg_errors = [avg_errors[j]+errors[j] for j in range(len(errors))]
    # plot the misclassifications vs num components
    plt.plot(num_components, errors)

avg_errors = [x / ITERATIONS for x in avg_errors]
print(avg_errors)
plt.show()
