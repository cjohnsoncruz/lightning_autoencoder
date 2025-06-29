"""
External functions used in run_autoencoder_lightning_v3.ipynb collected from their source locations.
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import itertools
import matplotlib
import seaborn as sns
from datetime import datetime
from sns_plotting_config import *
from ax_modifier_functions_cloud import *
stage_col = 'task_phase_vec'

# From helper_functions.py
def make_folder(folder_name):
    """Create a folder if it doesn't exist."""
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name, exist_ok=True)
            print(f"Created folder: {folder_name}")
        return folder_name
    except (PermissionError, FileNotFoundError) as e:
        print(f"Error creating folder '{folder_name}': {str(e)}")
        print("Try using a different path or checking permissions")
        return None

def make_figtype_subfolders(folder_name):
    """Create subfolders for different figure types."""
    subfolder_dict = {filetype:f"{folder_name}\\{filetype}\\" for filetype in ['svg','pdf','png','eps']}
    for ftype, subf in subfolder_dict.items():
        make_folder(subf)
    return subfolder_dict

def save_fig_in_main_fig_dir(fig_obj, fig_name, folder_key, filetypes_to_save, **kwargs):
    """
    Given a list of strings, dict of folder names, and template to fill, save N figures with N filetypes
    subfolder_dict- dict where key = filetype, value = folder name to save within current folder
    filetypes to save- list of strings whre elemtns = file exntensions (e.g. pdf, svg, png)
    """
    from helper_functions import fig_save_dir,supp_fig_save_dir
    main_fig_dict = {**fig_save_dir, **supp_fig_save_dir} #combine regular and supp fig dict
    folder_name = main_fig_dict[folder_key]
    subfolder_dict = make_figtype_subfolders(folder_name)#make subfolders
    for filetype in filetypes_to_save: #for each filetype, make a custom string location 
        save_loc = subfolder_dict[filetype] + fig_name+ "." + filetype
        fig_obj.savefig(save_loc, **kwargs)
        print(f"saved {save_loc}")

def save_plot_record_as_csv_txt(posthoc_df:pd.DataFrame,
                                folder_pref:str, #tag for later recorveyr fo folder information 
                                fig_name:str, 
                                csv_folder_most_recent:str= "", 
                                csv_folder_current_run:str="",
                                csv_suffix:str="_posthoc table_",
                                txt_suffix:str="results_text"):
    '''
    To save CSV + TXT records from the posthoc testing done for current figure.
    Saves in 1) 'csv folder most recent' which is constantly added to, and 2) csv folder current run, updated each day
    # params: csv_suffix and txt_suffix are appended to their respective filetypes
    return: none 
    '''
    ## get/import params and add to posthoc_df
    date_tag = get_datetag()
    posthoc_df['date_tag'] = date_tag
    posthoc_df['fig_name'] = fig_name
    posthoc_df['fig_num'] = folder_pref
    #save posthoc CSV
    if csv_suffix is not None:
        csv_name = "_".join([folder_pref, fig_name , csv_suffix , f"{date_tag}.csv"])
        save_csv_to_analysis_storage(posthoc_df.assign(fig_name=fig_name), csv_name, csv_folder_most_recent, csv_folder_current_run)
    else: 
        print("No csv suffix provided, not saving csv")
    if txt_suffix is not None:
        text_name = "_".join([folder_pref, fig_name , txt_suffix ,f"{date_tag}.txt"])
        #save as txt
        write_posthoc_to_txt_clean(posthoc_df,csv_folder_current_run+ text_name)
        write_posthoc_to_txt_clean(posthoc_df, csv_folder_most_recent + text_name)
    else:
        print("No text suffix provided, not saving text")

def get_datetag():
    """Get date tag in format day_month_year"""
    from datetime import datetime
    curr_time = datetime.now()
    date_tag = "_".join([curr_time.strftime('%d'),curr_time.strftime('%h'),curr_time.strftime('%Y')])
    return date_tag

def save_csv_to_analysis_storage(csv_to_save: pd.DataFrame, csv_name: str, csv_folder_latest: str, csv_folder_current_run: str) -> None:
    """
    Save a CSV file to both the latest analysis folder and the current run's analysis folder.
    """
    for folder in [csv_folder_latest, csv_folder_current_run]:
        os.makedirs(folder, exist_ok=True)
        csv_to_save.to_csv(os.path.join(folder, csv_name))
        print(f"Saved {csv_name} to {folder}")

def write_posthoc_table_to_txt(posthoc_table: pd.DataFrame, output_file: str) -> None:
    """
    Write a posthoc table to a text file, with one comparison per line.
    """
    with open(output_file, "w") as file:
        for _, row in posthoc_table.iterrows():
            file.write(f"Comparison: {row.get('comparison', 'N/A')}\n")
            file.write(f"Test: {row.get('test_type', 'N/A')}\n")
            significant = "Yes" if row.get('significant', False) else "No"
            file.write(f"Significant: {significant}\n")
            file.write(f"p-value: {row.get('p-val', 'N/A')}\n")
            file.write(f"Effect size: {row.get('effect_size', 'N/A')}\n")
            file.write("\n")
        print(f"Wrote results to {output_file}")
        
def make_chunked_iterator_of_ranges(start,end, n_batches):
    batch_size = end//n_batches #divide len of vector into N batches
    start_val_range = np.arange(start, end + batch_size, batch_size)
    end_val_range = np.arange(start + batch_size, end + batch_size, batch_size)
    start_end_val_list = [np.arange(x,y) for (x,y) in zip(start_val_range, end_val_range)]
    #avoid last batch being extralong
    start_end_val_list[-1] = np.arange(start_end_val_list[-1][0], end)
    print(f"N_batches: {n_batches}")
    print(f"Batch_size: {batch_size}")
    return start_end_val_list

#from preprocess.py
def get_numeric_cols_timeseries(input_df, numeric_sep):
    local_numeric_col = input_df.columns[input_df.columns.str.contains(numeric_sep)]
    return local_numeric_col
    
def get_unit_mean_timeseries_by_phase(time_series_df,cols_to_avg_over = None, groupby_list = None, numeric_col_wide = None):
    ##default args 
    if cols_to_avg_over== None:
        cols_to_avg_over = ['trial_num']
    if groupby_list== None: 
        groupby_list = ['name', 'neuron_ID','geno_day', 'task_phase_vec']
    ##main logic
    id_cols = time_series_df.columns[~time_series_df.columns.isin(numeric_col_wide)]
    temp_id_cols = [col for col in id_cols.tolist() if col not in groupby_list] #tewmp ID cols are for just maintaining information about activation
    groupby_func_agg = {**{col: 'mean' for col in numeric_col_wide}, **{id_col: 'first' for id_col in temp_id_cols if id_col not in cols_to_avg_over}}
    cell_avg_stage_tseries =time_series_df.groupby(by =groupby_list).agg(groupby_func_agg).reset_index()

    cell_avg_stage_tseries['max_val']= cell_avg_stage_tseries.loc[:,[c for c in cell_avg_stage_tseries.columns if "to" in c]].max(axis = 1)
    cell_avg_stage_tseries['max_val_tbin']= cell_avg_stage_tseries.loc[:,[c for c in cell_avg_stage_tseries.columns if "to" in c]].idxmax(axis = 1)
    cell_avg_stage_tseries['mean_rate'] = cell_avg_stage_tseries[[c for c in numeric_col_wide if '-' not in c]].mean(axis = 1) #mean rate is post outcome only
    return cell_avg_stage_tseries

def get_subject_stage_info_df(trial_tseries):
    #TO- create output DF with information about the # of enrihced units by stage, number of trials per stage, etc
    # get subj level dfs 
    trial_list_by_dataset = get_trial_num_in_phase_by_dataset( trial_tseries)
    e_unique_ID_subj = get_ID_enriched_units_by_phase(trial_tseries, 'unique_ID', 'enriched_in_phase')
    n_units_by_subject =  trial_tseries.groupby(by = ['geno_day', 'name'])['unique_ID'].nunique().reset_index().rename({'unique_ID': 'num_units'}, axis = 1)
    #merge all subj level dfs
    subject_stage_info_df = e_unique_ID_subj.merge(n_units_by_subject, on = ['geno_day', 'name'], how = 'left').merge(
        trial_list_by_dataset, on = ['geno_day', 'name', 'task_phase_vec'], how = 'left')
    subject_stage_info_df['over_5'] = subject_stage_info_df.apply(lambda x: [y > 5 for y in x['trial_num']], axis = 1)
    return subject_stage_info_df

def get_trial_num_in_phase_by_dataset(tseries_df):
    trial_list_by_dataset = tseries_df.groupby(['name', 'geno_day','task_phase_vec'])['trial_num'].unique().reset_index()
    trial_list_by_dataset['count_of_trials'] =trial_list_by_dataset.trial_num.apply(lambda x: len(x))
    return trial_list_by_dataset

def get_ID_enriched_units_by_phase(df_with_enrichment, ID_col, bool_col_tseries_enriched):
    #bool_col_tsewries_enriched is a boolean col of df_with_enrichment that says if cell N in trial X from phase P is enriched in phase P
    df_only_enrich_in_curr_phase_tseries = df_with_enrichment[df_with_enrichment[bool_col_tseries_enriched]] #index into tseries df with bool col
    enrich_unit_ID_by_name_df = df_only_enrich_in_curr_phase_tseries.groupby(['name', 'geno_day','task_phase_vec'])[ID_col].unique().reset_index()
    enrich_unit_ID_by_name_df['num_enriched_units'] = enrich_unit_ID_by_name_df[ID_col].apply(lambda x: len(x))
    return enrich_unit_ID_by_name_df

# From phase_enrichment_timeseries/phase_enrich_timeseries_decode.py

def fast_pack_data_local(class_0_mat, class_1_mat,class_0_value, class_1_value):
    #TO- given 2 objects containg ['matrix', 'labels'] fields, join each field's content together and export (Transposing for easier use later)
    #returns numpy array which = transpose of concat. data matrix and labels
    concat_matrix = class_0_mat.join(class_1_mat, how = 'inner', lsuffix = '_c0', rsuffix = '_c1') #use inner to avoid having missing cells in a condition if dataset missing that phase's trials
    #outer keeps empty cells from another dataset, which is OBVIOUS clue if condition 1 is e.g. missing early IA error cells from some datasets
    class_label_0 = make_class_label_vector(class_0_mat, class_0_value)
    class_label_1 = make_class_label_vector(class_1_mat, class_1_value)
    concat_labels =  np.concatenate([class_label_0,class_label_1 ]) #concat class 0 and 1 label vector and optionally add as final row of matrix (to shuffle it exactly the same)
    return concat_matrix, concat_labels #transpose matrix and labels to avoid having to ranspose later
#reample data
def make_class_label_vector(class_matrix, class_val):
    class_labels = np.repeat(class_val, class_matrix.shape[1])
    return class_labels

def return_train_test_tensorDataset(input_data, input_labels:np.array, shuffle:bool = True, test_on_train:bool = True):
    '''TO- given numpy matrices, and optional shuffle, return 2 pytorch test and train datasets in tensor format'''
    #moved from outside func to inside
    input_data = input_data.values.T #.T.values of concat matrix is to reshape mat into what scikit wants from you
    #get indices
    assert input_data.shape[0] > 0
    rng = np.random.default_rng()
    n_samples = input_data.shape[0]
    input_data_index =np.arange(0,n_samples) 
    permuted_index = rng.permutation(input_data_index)

    if test_on_train:
        if shuffle:
            train_index = permuted_index
            test_index = permuted_index
        else: #non shuffle, test set = train set
            train_index = input_data_index
            test_index = train_index
    else: #if test not == train
        if shuffle:
            train_index = permuted_index[0:n_samples//2]
            test_index = permuted_index[n_samples//2:n_samples]
        else:
            train_index = input_data_index[0:n_samples//2]
            test_index =  input_data_index[n_samples//2:n_samples]
        
    ##extract data of interest 
    train_data= torch.tensor(input_data[train_index,:], dtype=torch.float32)
    train_labels= torch.tensor(input_labels[train_index], dtype=torch.float32)
    
    test_data = torch.tensor(input_data[test_index,:], dtype=torch.float32)
    test_labels = torch.tensor(input_labels[test_index], dtype=torch.float32)
    ## consolidate into TEnsorDataset objects
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    
    return train_dataset, test_dataset


def save_fig_as_filetype_list(fig_obj, fig_name_template, subfolder_dict, filetypes_to_save):
    """ Given a list of strings, dict of folder names, and template to fill, save N figures with N filetypes
    subfolder_dict- dict where key = filetype, value = folder name to save within current folder
    filetypes to save- list of strings whre elemtns = file exntensions (e.g. pdf, svg, png )"""
    #NEW_ add datetime tag
    date_tag = "_".join([datetime.now().strftime('%d'),datetime.now().strftime('%h'),datetime.now().strftime('%Y')])
    # add date-tag annot
    # fig_obj.get_axes()[0].annotate(f"made-{date_tag}", (0.25,1.005), (0, 0),  fontsize = 5, xycoords='figure fraction', textcoords='offset points',annotation_clip = False, ha= 'left', va='top')

    for fig_type in filetypes_to_save:
        fig_obj.savefig(subfolder_dict[fig_type] + fig_name_template + f".{fig_type}")

def save_metadata_record(last_run_metadata_record:pd.DataFrame, metadata_record_filename:str, save_loc:str):
    #write filename to dF
    
    if not os.path.exists(metadata_record_filename):    #create locally then save
        print(f"creating autoencoder versioning file: {metadata_record_filename}")
        last_run_metadata_record.to_csv(metadata_record_filename,index = False)
    else: #load existing and add to it 
        print(f"Loading autoencoder versioning file: {metadata_record_filename}")
        metadata_record= pd.read_csv(metadata_record_filename)
        metadata_record = pd.concat([metadata_record, last_run_metadata_record]) #join prev df with latest DF
        #save multiple backups 
        # metadata_record.to_csv(metadata_record_filename) 
        metadata_record.to_csv(save_loc + metadata_record_filename)
        metadata_record.to_csv(csv_folder_most_recent +metadata_record_filename)
        metadata_record.to_csv(csv_folder_current_run + metadata_record_filename)


# Define the function to apply to each group
def compute_dbi(group):
    # Only compute DBI if there are at least 2 unique labels
    if group['labels'].nunique() < 2:
        print(f"Error:less than 2 unique labels found")
        return pd.Series({'Davies-Bouldin index': np.nan})
    
    X = group[['embed_1', 'embed_2']].to_numpy()
    y = group['labels'].to_numpy()
    dbi = davies_bouldin_score(X, y)
    return pd.Series({'Davies-Bouldin index': dbi})



def plot_class_scatter_in_latent(fig, ax_array, df_comparison, compared_col, ens_col, ensemble_subset, comparison,
                                 make_legend = True, samples_to_plot:int = 500, use_stage_colors= True, **kwargs):

    #set logic for colors to use
    if use_stage_colors:
        scatter_palette=  [stage_palette_dict[s] for s in ensemble_subset]
    else: 
        scatter_palette=['red','blue']
    ##set default params for plots:
    plot_defaults = dict(alpha =.5, linewidth = 0.5)
    plot_kwargs = {**plot_defaults, **kwargs}
    comparison_clean = comparison.replace("_v_", " & ").replace("_", " ")

    for c_count, ensemble in enumerate(ensemble_subset):#Loop over all combinations    
        for g_count, g in enumerate(geno_order):    ##COL ITERATION
            plot_df = df_comparison.loc[(df_comparison[ens_col]== ensemble) & (df_comparison.geno == g),:].copy()
            legend_bool = {True: (g_count == len(geno_order)-1) & (c_count == len(ensemble_subset)-1),
                               False: False}[make_legend]
            ##optionally pull subset of cells
            input_data_index =np.arange(0, plot_df.shape[0]) 
            permuted_index =  np.random.default_rng().permutation(input_data_index)
            cell_subset = permuted_index[:samples_to_plot]
            plot_df = plot_df.iloc[cell_subset,:]
            #plot latent space  projection
            ax = ax_array[c_count,g_count]
            sns.scatterplot(data = plot_df,ax = ax, x= "embed_1", y= "embed_2", legend = legend_bool,
                            s= 3, hue_order = set(plot_df.labels.unique()),hue= 'labels', palette=scatter_palette, **plot_kwargs)
            #make ax mods
            for coll in ax.collections:
                coll.set_edgecolors(coll.get_facecolors().copy())# Restore the edge colors using the saved values
                coll.set_facecolors("none")# Remove the face color by setting it to "none"
            
            ax_title =  dict(label=f"{g}", pad = 2)
            set_ax_title_xlabel_ylabel(ax, label_dict = {'title': ax_title, 
                                                         'xlabel': 'Latent Dim. 1',
                                                         'ylabel': 'Latent Dim. 2',
                                                         'xticks':[], 'yticks':[] })    
    ##NEW- 5.9.25- add row title to latent space    #set titles for pseudo congig 
    row_titles = [f"{e.replace("_", " ")} ensemble: {comparison_clean} latent space" for e, test in zip(
        ensemble_subset, [comparison_clean,comparison_clean])]
    add_ax_array_row_title(fig, row_titles, ax_array)
    
    if make_legend:#new- given removal of suptitle, now get text figure
        fig.canvas.draw() # force a draw so that the Text has a renderer
        renderer = fig.canvas.get_renderer()
        text_objs = [x for x in fig.get_children() if isinstance(x,matplotlib.text.Text)]
        bbox_fig = text_objs[0].get_window_extent(renderer).transformed(fig.transFigure.inverted()) # 3. get the bbox in display (pixel) coords
        hand, labels = ax_array.flat[-1].get_legend_handles_labels()
        ax_array.flat[-1].get_legend().remove() 
        legend_y = bbox_fig.y1 + 0.025
        fig.legend(handles = hand, labels = [x.replace("_", " ") for x in labels ], 
                   bbox_to_anchor=(.90, legend_y), loc = 'center right', frameon = False, title = None, ncol=2)   
    #last step, 
    for ax in ax_array.flat:# force each subplot’s BOX to be square:
        ax.set_box_aspect(1)
        ax.xaxis.labelpad= 2
        ax.yaxis.labelpad= 2

def resample_ensemble_stage_activity(class_matrix_store,
                                     n_frames_to_draw:int=500,
                                     ensemble:str = "",
                                     geno:str="",
                                     stages_to_resample:list  = ['Early_IA_Correct', 'Early_RS_Correct', 'Early_IA_Error', 'Early_RS_Error'],
                                     **kwargs):
    '''To iterate over specified stages and return dict containing N resampled frame for each stage. Requires 'resample_into_class_matrix'
    '''
    geno_ensemble_activity_in_stage = {}
    for class_name in stages_to_resample: #store the versions for this bootstrap run
        class_mat = resample_into_class_matrix(class_matrix_store,n_frames_to_draw,ensemble, class_name, geno)
        geno_ensemble_activity_in_stage[class_name] = class_mat
        
    return geno_ensemble_activity_in_stage


def resample_concat_dict_of_activity_mats_w_subsample(dict_of_activity_dfs, num_resample,local_rng, subsample_range):
    """TO: given a dict of activity matrices where rows = units, cols = frames and scalar N, resample it N times into a list. 
    Each list elem = activity from a diff subj (to allow for varying len datasets)"""
    #pandas dataframe is INCLUSIVE of the end value, so you are including it in the subsample
    resampled_dict_list =[resample_unit_timeseries_df(d.loc[:,subsample_range], num_resample,local_rng) for d in dict_of_activity_dfs.values()]
    resampled_df = pd.concat(resampled_dict_list, axis = 0)
    return resampled_df


def resample_into_class_matrix(class_matrix_store,n_resample, ensemble_name, class_name, geno_day_curr, subsample_range = []):
    local_rng = np.random.default_rng() #create a Generator instance with default_rng
    activity_dict = class_matrix_store[ensemble_name][class_name][geno_day_curr]
    ## add logic for ability to subsample what frame you actually use 
    if len(subsample_range) == 0:
        subsample = False
    else:
        subsample = True
    if subsample:
        class_matrix = resample_concat_dict_of_activity_mats_w_subsample(activity_dict,n_resample,local_rng, subsample_range)
    else:
        class_matrix = resample_combine_dict_of_activity_dfs(activity_dict, n_resample,local_rng)
    return class_matrix

def resample_combine_dict_of_activity_dfs(dict_of_activity_dfs, num_resample,local_rng):
    """TO: given a dict of activity matrices where rows = units, cols = frames and scalar N, resample it N times into a list. 
    Each list elem = activity from a diff subj (to allow for varying len datasets)"""

    resampled_dict_list =[resample_unit_timeseries_df(d, num_resample,local_rng) for d in dict_of_activity_dfs.values()]
    #NO LONGER RETURNS TUPLES 9/26/24/ #above returns list of tuples (resample df, resample IDs)
    # resampled_df = pd.concat([e for e in resampled_dict_list], axis = 0)
    resampled_df = pd.concat(resampled_dict_list, axis = 0)
    # class_idx = [e[1] for e in resampled_dict_list]
    return resampled_df

def resample_unit_timeseries_df(input_timeseries_df, num_resample,local_rng):
    #TO- retain original tseries index, but join to newly resample df
    resample_matrix = resample_matrix_rows_w_replace(input_timeseries_df.values, num_resample, local_rng) 
    resample_unit_df = pd.DataFrame(data = resample_matrix, index = input_timeseries_df.index)
    return resample_unit_df

def resample_matrix_rows_w_replace(input_matrix, num_resample,local_rng):
    """ returns resampled matrix, and indices used to generate resample. feed in pre-created resample """
    array_rows, array_cols = input_matrix.shape #get dims of input_matrix #assume that array_rows = num_units
    num_units = array_rows #assme that the num rows in the tseries = num units you sample from in dataset
    resample_idx = local_rng.choice(array_cols, size = (num_units,num_resample),
                                     replace = True) #output shape: (num_units,num_resample)
    resampled_matrix = np.take_along_axis(input_matrix, resample_idx, axis=1) #axis = 1 means slices  in rows
    #Take values from input mat by matching 1d index vals in data slices.
    return resampled_matrix