import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from os import listdir
from multiprocessing import Pool
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing import event_accumulator

# Helpful links:
# https://stackoverflow.com/questions/41074688/how-do-you-read-tensorboard-files-programmatically

# Workflow:
# 1. Specify which runs and which graph I want to plot
# 2. Specify which agent's run you are interested in plotting
# 3. Specify if you want to average the values if more than one run is selected
# 4. Plot lables and axis lables, legend
# 5. Smoothing factor
# 6. output save file format

def worker(tb_tuple):
    """Make a dict out of the parsed, supplied lines"""
    ea = event_accumulator.EventAccumulator(tb_tuple[0],
        size_guidance={ # see below regarding this argument
        event_accumulator.COMPRESSED_HISTOGRAMS: 500,
        event_accumulator.IMAGES: 4,
        event_accumulator.AUDIO: 4,
        event_accumulator.SCALARS: 0,
        event_accumulator.HISTOGRAMS: 1,
    })
    ea.Reload()
    _, _, vals = zip(*ea.Scalars(tb_tuple[1]))
    return vals

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

def smooth_mpc(args):  # Weight between 0 and 1
    scalars, weight = args[0], args[1]
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

def plotData(fig, coordinate, data, axisLabels, title, avg=False, color='b', labels=None, legend=False, savefile=None):
    # fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(coordinate) #nrow, ncol, index starting at top left
    ax.set_xlabel(axisLabels[0])
    ax.set_ylabel(axisLabels[1])
    # ax.set_title(title)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    if avg:
        all_vals = np.array(data)
        mean_vals = np.mean(all_vals, axis=0)
        std_vals = np.std(all_vals, axis=0)

        ax.plot(mean_vals, label=labels, color=color)
        step_nums = np.linspace(1, len(mean_vals), len(mean_vals))
        ax.fill_between(step_nums, mean_vals-std_vals, mean_vals+std_vals, color=color, alpha=0.2)
    else:
        for i in range(len(data)):
            label = labels[i] if labels else None
            ax.plot(data[i], label=label, color=colors[i])
    if legend:
        ax.legend()
    ax.grid(linestyle='--')
    if savefile:
        fig.savefig(savefile)
    return fig

if __name__ == '__main__':
    # baseline_root_path = '/home/s6pereir/multirun_results/general_stkl_v2_results/August_2019/' 
    baseline_root_path = '/home/s6pereir/multirun_results/seminar_runs/'
    scenario_dirs = [
                     ['variable_fgs/September_2019/maddpg_particle_tag_stkl_v3_fgs_sig_r1_25_15-36_5632',
                      'variable_fgs/September_2019/maddpg_particle_tag_stkl_v3_fgs_sig_r2_30_17-13_85zx',
                      'variable_fgs/September_2019/maddpg_particle_tag_stkl_v3_fgs_sig_r4_30_17-13_cmjd'],
                     ['baseline_results/September_2019/maddpg_particle_env_tag_128_r2_30_22-49_k3bq',
                      'baseline_results/September_2019/maddpg_particle_env_tag_128_r3_30_22-48_134o',
                      'baseline_results/September_2019/maddpg_particle_env_tag_128_r5_30_22-49_e899'],
                     ['orig_comm/September_2019/maddpg_particle_tag_baseComm_128_r1_18_22-03_vfb5',
                      'orig_comm/September_2019/maddpg_particle_tag_baseComm_128_r3_18_22-07_zfo4',
                      'orig_comm/September_2019/maddpg_particle_tag_baseComm_128_r4_18_22-11_sw5b']]
                    #  ['variable_fgs/September_2019/maddpg_particle_tag_stkl_v3_fgs_sig_r1_25_15-36_5632',
                    #   'variable_fgs/September_2019/maddpg_particle_tag_stkl_v3_fgs_sig_r2_30_17-13_85zx',
                    #   'variable_fgs/September_2019/maddpg_particle_tag_stkl_v3_fgs_sig_r4_30_17-13_cmjd'],
                    #  ['variable_fgs/September_2019/maddpg_particle_tag_stkl_v3_fgs_step_r1_25_15-48_pk7v',
                    #   'variable_fgs/September_2019/maddpg_particle_tag_stkl_v3_fgs_step_r3_30_16-52_z00t',
                    #   'variable_fgs/September_2019/maddpg_particle_tag_stkl_v3_fgs_step_r4_30_16-52_6wlx',
                    #   'variable_fgs/September_2019/maddpg_particle_tag_stkl_v3_fgs_step_r5_30_16-52_puzx']]

                    #   'baseline_results/August_2019/maddpg_particle_env_tag_r2_24_19-24_w2yr',
                    #   'baseline_results/August_2019/maddpg_particle_env_tag_r4_24_19-27_q3ay',
                    # 'stkl_v3/September_2019/maddpg_particle_tag_stkl_v3_128_fgs025_r1_18_17-24_4f7r',
                    # 'stkl_v3/September_2019/maddpg_particle_tag_stkl_v3_128_fgs075_r3_25_20-11_h6pe'
                    # 'orig_comm/September_2019/maddpg_particle_tag_baseComm_128_r2_18_22-06_7h5f', 
                    # 'orig_comm/September_2019/maddpg_particle_tag_baseComm_128_r5_18_22-13_dbt4'
    plot_list = ['gym_highway/tb/ro/ag3_return_history','gym_highway/tb/ro/ag0_return_history']
    # plot_list = ['gym_highway/tb/ro/ag3_return_history','gym_highway/tb/ro/ag0_return_history']
    # plot_labels = ['baseComm_avg', 'standard_avg']
    # plot_labels = ['exp', 'linear', 'log', 'sig', 'step']
    # plot_labels = ['0.0', '0.25', '0.5', '0.75', '1.0']
    # plot_labels = ['MADDPG+Comm','SMARL']
    # plot_labels = ['r1','r2','r3','r4','r5']
    # plot_labels = ['Agent 0','Adversary 1','Adversary 2']
    # plot_labels = ['Leader','Follower','Follower']
    plot_labels = ['SMARL','MADDPG','MADDPG + Comm']
    # plot_labels = ['r1_000','r2_000','r3_000','r1_100','r2_100','r3_100']
    # plot_labels = ['Prey','Predators', 'Predator 2', 'Predator 3']
    agents = []
    avg = True
    plot_title = None
    x_label = "Episodes"
    # y_label = ["Leader avg. reward", "Follower 1 avg. reward", "Follower 2 avg. reward"]
    # y_label = ["Leader reward", "Follower 1 reward",  "Follower 2 reward", "Followers reward"]
    # y_label = ["Agent 0 reward", "Adversary reward", "Adversary reward"]
    y_label = ["Prey avg. reward", "Predator avg. reward", "Predator avg. reward", "Predator avg. reward"]
    add_legend = True
    smooth_factor = 0.999
    output_file = "seminar_outputs/particle_tag_128_avg_smarl_vs_maddpg_vs_maddpgComm_0999.pdf"

    # scenario = "maddpg_driving_BG_baseComm_r1_22_15-32_o33k"
    # baseline_runs = [os.path.join(baseline_root_path, f) for f in listdir(baseline_root_path) if scenario in f]
    scenario_fullpaths = []
    for scenario in scenario_dirs:
        scenario_path = [os.path.join(baseline_root_path, f) for f in scenario]
        tb_paths = []
        for f in scenario_path:
            sub_dir = f+'/tb/'
            tb_outs = [os.path.join(sub_dir, r) for r in listdir(sub_dir)]
            tb_paths.append(tb_outs[0])
        scenario_fullpaths.append(tb_paths)

    print("scenario_fullpaths")
    print(scenario_fullpaths)

    start_time = time.time()
    numthreads = 6
    pool = Pool(processes=numthreads)

    # fig = plt.figure(figsize=[6, 6])
    fig = plt.figure(figsize = (12,6))
    # fig = plt.figure(figsize = (18,6))
    coordinate = 121
    # plt.title(plot_title)
    # plt.ylabel(y_label)
    # plt.xlabel(x_label)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    index = 0
    for p in range(len(plot_list)):
        plot_item = plot_list[p]
        do_save = p == len(plot_list)-1 
        # coordinate += p
        for i in range(len(scenario_fullpaths)):
            scenarios = scenario_fullpaths[i]
            scenarios = [(s, plot_item) for s in scenarios]
            all_vals = pool.map(worker, scenarios)
            # scenarios_args = [(vals, smooth_factor) for vals in all_vals]
            # smooth_vals = pool.map(smooth_mpc, scenarios_args)
            smooth_vals = [smooth(vals, smooth_factor) for vals in all_vals]
            # smooth_vals = all_vals
            label = plot_labels[i] if avg else plot_labels
            # index += 0 if avg else 1
            # coordinate += i
            save_file = output_file if do_save and (i == len(scenario_fullpaths)-1) else None
            plotData(fig, coordinate+index, smooth_vals, [x_label, y_label[p]], plot_title, avg=avg, color=colors[i], labels=label, legend=True, savefile=save_file)

            # Leaders and followers on same plot
            # label = plot_labels[p] if avg else plot_labels
            # plotData(fig, coordinate+index, smooth_vals, [x_label, y_label[p]], plot_title, avg=avg, color=colors[p], labels=label, legend=True, savefile=save_file)
        index = min(index+1, len(plot_list)-1)

    print("Processing time: ", time.time()-start_time)

    exit(0)

    event_accumulators = []
    for tbo in scenario_fullpaths[:5]:
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

    # event_accumulators = pool.map(worker, baseline_tb_output_files)

    print("Processing time: ", time.time()-start_time)
    print("Total event accumulators: ", len(event_accumulators))

    # print(event_accumulators[0].scalars.Keys())

    # exit(0)

    plt.figure(figsize=[6, 6])
    plt.title(plot_title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    colors = ['r', 'g', 'b','c','y']
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for plot_item, p_label in plot_list:
        all_vals = []
        for i in range(len(event_accumulators)):
            ea = event_accumulators[i]
            w_times, step_nums, vals = zip(*ea.Scalars(plot_item))
            # vals_df = pd.DataFrame(np.array(vals))
            smooth_val = smooth(vals, smooth_factor)
            all_vals.append(smooth_val)
            # plt.plot(vals_df[0], 'lightblue', smooth_val, 'b')

            # plt.plot(smooth_val, label=plot_labels[i])
        if average_vals:
            all_vals = np.array(all_vals)
            mean_vals = np.mean(all_vals, axis=0)
            std_vals = np.std(all_vals, axis=0)
            # mean_vals = avg_vals/len(event_accumulators)
            plt.plot(mean_vals, label='avg_value')
            plt.fill_between(step_nums, mean_vals-std_vals, mean_vals+std_vals, color='blue', alpha=0.1)
        if add_legend:
            plt.legend()
        # plt.show()
    plt.grid(linestyle='--')
    plt.savefig(output_file)
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

    # plt.figure(figsize=[6, 6])
    # plt.plot(vals_df[0], 'lightblue', smooth_val, 'b')
    # plt.ylabel('Values')
    # plt.xlabel('Steps')
    # plt.savefig("smooth_0.9.svg")

    # ea.Scalars('gym_highway/tb/ro/return_history').Keys()
    ea.scalars.Keys()
    all_scalar_events = ea.scalars.Items('gym_highway/tb/ro/return_history')

    all_steps_per_key = [[scalar_events.step, scalar_events.value, scalar_events.wall_time] for scalar_events in all_scalar_events]
    all_steps_per_key


    # function to plot data points
    # def plotData(data, axisLabels, title):
    #     fig = plt.figure(figsize = (8,8))
    #     ax = fig.add_subplot(1,1,1) 
    #     ax.set_xlabel(axisLabels[0], fontsize = 15)
    #     ax.set_ylabel(axisLabels[1], fontsize = 15)
    #     ax.set_title(title, fontsize = 20)
    #     targets = [0,1,2,3,4]
    #     colors = ['r', 'g', 'b','c','y']
    #     for target, color in zip(targets,colors):
    #         indicesToKeep = data['class'] == target
    #         ax.scatter(data.loc[indicesToKeep, axisLabels[0]]
    #                 , data.loc[indicesToKeep, axisLabels[1]]
    #                 , c = color
    #                 , s = 50)
    #     ax.legend(targets)
    #     ax.grid()


    # function to plot images as points
    # def imscatter(x, y, image, axisLabels, title, ax=None, zoom=1):
    #     if ax is None:
    #         ax = plt.gca()
    #     x, y = np.atleast_1d(x, y)
    #     artists = []
    #     counter = 0
    #     for x0, y0 in zip(x, y):
    #         im = OffsetImage(image[counter], zoom=zoom, cmap=plt.cm.gray_r)
    #         ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
    #         artists.append(ax.add_artist(ab))
    #         counter += 1
    #     ax.update_datalim(np.column_stack([x, y]))
    #     ax.set_xlabel(axisLabels[0], fontsize = 15)
    #     ax.set_ylabel(axisLabels[1], fontsize = 15)
    #     ax.set_title(title, fontsize = 20)
    #     ax.autoscale()
    #     return artists


    # compute misclassificatoins ITERATIONS number of times to get an average
    # for i in range(ITERATIONS):
    #     errors = []
    #     for num_comp in num_components:
    #         # perform dimensionality reduction
    #         dimReducedDf = dimReduction(samples_std, num_comp)
    #         # split the data into 70% training and 30% test sets 
    #         X_train, X_test, y_train, y_test = train_test_split(dimReducedDf, dataB_gnd, test_size=0.3)
    #         # train using training sets, and predict labels of test set
    #         y_pred = gnb.fit(X_train.values, y_train).predict(X_test.values)
    #         # count the number of misclassified values
    #         errors.append((y_test.ravel() != y_pred).sum())
        
    #     # keep track of total number of misclassifications
    #     avg_errors = [avg_errors[j]+errors[j] for j in range(len(errors))]
    #     # plot the misclassifications vs num components
    #     plt.plot(num_components, errors)

    # avg_errors = [x / ITERATIONS for x in avg_errors]
    # print(avg_errors)
    # plt.show()
