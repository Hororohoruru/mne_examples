"""
MEG assignment from Parietal team for MNE

Author: Juan Jesus Torre
email: juanjesustorre@gmail.com

TODO:
    - Handle skipping steps if files are already computed
    - Add overwrite
    - Add verbose parameter
    - Overwrite the list with the data at each step so it does not blow memory
    - Check picks for ICA
"""

import argparse
import glob
import os

import mne

from IPython.terminal.embed import InteractiveShellEmbed
from joblib import Parallel, delayed
from mne.preprocessing import maxwell_filter

from assignment_utils import (get_data_paths, get_subject_ids, run_maxfilter, filter_data,
                              run_ica_correction, save_files)


parser = argparse.ArgumentParser(description='Parameters for the pipeline')
parser.add_argument('-t', '--type', metavar='FileType', type=str,
                    default='passive', choices=['passive', 'rest', 'task'],
                    help="File type to analyze for each subject. Choices: "
                         "%(choices)s. Default: %(default)s")
parser.add_argument('-lf', '--lowfreq', metavar='LowFrequency', type=int,
                    default=1, help="Low frequency for band-pass filter. "
                                    "Default: %(default)s")
parser.add_argument('-hf', '--hifreq', metavar='HighFrequency', type=int,
                    default=40, help="High frequency for band-pass filter. "
                                    "Default: %(default)s")
parser.add_argument('-p', '--power', metavar='PowerLine', type=int,
                    default=50, help="Frequency for notch filter harmonics."
                                     "Default: %(default)s")
parser.add_argument('-s', '--save', metavar='Save', type=bool,
                    default=True, choices=[True, False],
                    help="File type to analyze for each subject. Choices: "
                         "%(choices)s. Default: %(default)s")

args = parser.parse_args()
file_type = args.type
l_freq = args.lowfreq
h_freq = args.hifreq
power_line = args.power
save = args.save


# Some global arguments
root_dir = '/home/jtorretr/neuroimaging/eeg_data/neurosynchrone/data'
save_dir = '/home/jtorretr/neuroimaging/eeg_data/neurosynchrone/data/derivatives'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# First, we get the subject IDs and load their data
raw_names = glob.glob(os.path.join(root_dir, '*.set'))

# The keys of this dictionary are the subject IDs, and the values are preloaded raw files
raw_dict = {os.path.basename(file).split('.')[0]: mne.io.read_raw_eeglab(file, preload=True) for
            file in raw_names}

# Apply notch filter and band-pass filter to the data
# The resulting list consists of tuples with (sub_id, filter_raw) information
filter_raw_list = Parallel(n_jobs=len(raw_names))(delayed(filter_data)(
                           sub_id=sub_id,
                           raw=raw,
                           results_dir=save_dir) for sub_id, raw in raw_dict.items())

print("")
print(f'Notch filter at {power_line} and band pass filter between {l_freq} and {h_freq} '
      f'applied to all subjects')
print("")

# Subject by subject, enter interactive mode to visually inspect the data and mark bad channels,
# that will be excluded after you manually close the interactive console pressing Ctrl+D
for id, data in filter_raw_list:
    print(f"Interactive mode for sub {id}. Please check the data and mark bad channels "
          f"if necessary (adding them to data.info['bads'])")
    print("")

    shell = InteractiveShellEmbed()
    shell.enable_matplotlib()

    shell()

    data.pick_types(eeg=True, eog=True, ecg=True, stim=True, exclude='bads')

if save:
    save_files(filter_raw_list, save_dir, prefix='filtered', suffix=file_type)
    print("Saved notch and band pass filters applied")
    print("")

# # Now apply ICA correction for heartbeats and blinks
# ica_raw_list = Parallel(n_jobs=len(raw_names))(delayed(run_ica_correction)(
#                         sub_id=sub_id,
#                         raw=raw,
#                         results_dir=save_dir) for sub_id, raw in filter_raw_list)
#
# print("")
# print(f'ICA run for all subjects')
# print("")
#
# if save:
#     save_files(ica_raw_list, save_dir, prefix='ica_corrected', suffix=file_type)
#     print("Saved files with ICA correction applied for EOG and ECG artifacts")
#     print("")
