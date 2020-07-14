"""
MEG assignment from Parietal team for MNE

Author: Juan Jesus Torre
email: juanjesustorre@gmail.com

TODO:
    - Add argparser interface
    - Handle saving of intermediate files (maxfilter, band-pass + notch filter, etc)
    - Handle skipping steps if files are already computed
    - Add overwrite
    - Add verbose parameter
    - Overwrite the list with the data at each step so it does not blow memory
    - Check picks for ICA
"""

import os

import mne

from IPython.terminal.embed import InteractiveShellEmbed
from joblib import Parallel, delayed
from mne.preprocessing import maxwell_filter

from assignment_utils import (get_data_paths, get_subject_ids, run_maxfilter, filter_data,
                              run_ica_correction)

# Some global arguments
root_dir, save_dir, cal_fname, ctc_fname = get_data_paths()
file_type = 'passive'
l_freq = 1
h_freq = 40
power_line = 50

if file_type == 'passive':
    event_dict = {'Auditory 300Hz': 6,  # See trigger_codes.txt
                  'Auditory 600Hz': 7,
                  'Auditory 1200Hz': 8,
                  'Visual Checkerboard': 9}

# First, we get the subject IDs and load their data
sub_ids = get_subject_ids(root_dir=root_dir)
raw_names = [os.path.join(root_dir, f"{sub_id}/{file_type}/{file_type}_raw.fif")
             for sub_id in sub_ids]

# The keys of this dictionary are the subject IDs, and the values are preloaded raw files
raw_dict = {sub_id: mne.io.read_raw_fif(file, preload=True)
            for sub_id, file in zip(sub_ids, raw_names)}

# The resulting list consists of tuples with (sub_id, sss_raw) information
sss_raw_list = Parallel(n_jobs=len(sub_ids))(delayed(run_maxfilter)(
                        sub_id=sub_id,
                        raw=raw,
                        cal_fname=cal_fname,
                        ctc_fname=ctc_fname,
                        results_dir=save_dir) for sub_id, raw in raw_dict.items())

print("")
print('Maxfilter passed on all subjects')
print("")

# Subject by subject, enter interactive mode to visually inspect the data and mark bad channels,
# that will be excluded after you manually close the interactive console pressing Ctrl+D
for id, data in sss_raw_list:
    print(f"Interactive mode for sub {id}. Please check the data and mark bad channels "
          f"if necessary (adding them to data.info['bads'])")
    print("")

    shell = InteractiveShellEmbed()
    shell.enable_matplotlib()

    shell()

    data.pick_types(meg=True, eog=True, stim=True, ecg=True, exclude='bads')

# Apply notch filter and band-pass filter to the data
# The resulting list consists of tuples with (sub_id, filter_raw) information
filter_raw_list = Parallel(n_jobs=len(sub_ids))(delayed(filter_data)(
                           sub_id=sub_id,
                           raw=raw,
                           results_dir=save_dir) for sub_id, raw in sss_raw_list)

print("")
print(f'Notch filter at {power_line} and band pass filter between {l_freq} and {h_freq} '
      f'applied to all subjects')
print("")

# Now apply ICA correction for heartbeats and blinks
ica_raw_list = Parallel(n_jobs=len(sub_ids))(delayed(run_ica_correction)(
                        sub_id=sub_id,
                        raw=raw,
                        results_dir=save_dir) for sub_id, raw in filter_raw_list)

print("")
print(f'ICA run for all subjects')
print("")
