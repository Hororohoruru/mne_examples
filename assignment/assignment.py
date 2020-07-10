"""
MEG assignment from Parietal team for MNE

Author: Juan Jesus Torre
email: juanjesustorre@gmail.com
"""

import os

import mne

from IPython import embed
from joblib import Parallel, delayed
from mne.preprocessing import maxwell_filter

from assignment_utils import get_data_paths, get_subject_ids, run_maxfilter, filter_data


root_dir, save_dir, cal_fname, ctc_fname = get_data_paths()
file_type = 'passive'

sub_ids = get_subject_ids(root_dir=root_dir)
raw_names = [os.path.join(root_dir, f"{sub_id}/{file_type}/{file_type}_raw.fif")
             for sub_id in sub_ids]

# The keys of this dictionary are the subject IDs, and the values are preloaded raw files
raw_dict = {sub_id: mne.io.read_raw_fif(file, preload=True)
            for sub_id, file in zip(sub_ids, raw_names)}

# The resulting list consists of tuples with (sub_id, sss_raw) information
sss_raw_list = Parallel(n_jobs=len(sub_ids))(delayed(run_maxfilter)(
                        raw=raw,
                        cal_fname=cal_fname,
                        ctc_fname=ctc_fname,
                        sub_id=sub_id,
                        results_dir=save_dir) for sub_id, raw in raw_dict.items())

print("")
print('Maxfilter passed on all subjects')
print("")

for id, data in sss_raw_list:
    print(f"Interactive mode for sub {id}. Please check the data and mark bad channels"
          f"if necessary (adding them to data.info['bads'])")
    print("")

    # Embed an Ipython terminal
    embed()

    data.pick_types(meg=True, eeg=True, exclude='bads')
    filter_raw = filter_data(data, id)

    embed()