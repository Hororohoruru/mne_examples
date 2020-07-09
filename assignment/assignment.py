"""
MEG assignment from Parietal team for MNE

Author: Juan Jesus Torre
email: juanjesustorre@gmail.com
"""

import os

import joblib
import mne
from mne.preprocessing import maxwell_filter

from assignment.assignment_utils import get_data_paths, get_subject_ids


def run_maxwell(raw_file):
    """Run Maxwell filter on a raw file"""

    sss_raw = maxwell_filter(raw_file, origin='auto', calibration=cal_fname,
                             cross_talk=ctc_fname, st_duration=10)


root_dir, save_dir, cal_fname, ctc_fname = get_data_paths()

sub_ids = get_subject_ids()