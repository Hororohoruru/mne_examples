"""
Utils for the MEG assignment
"""

import os
import socket

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.io import read_raw_fif
from mne.preprocessing import maxwell_filter, ICA, create_ecg_epochs, create_eog_epochs


def get_subject_ids(root_dir, n_sub=None):
    """
    Find subject IDs in the root directory of the dataset

    Parameters
    ----------

    root_dir: str or pathlike object
              Directory containing the dataset

    n_sub: None or int, default None
           If not None, take the N first subjects from the list

    Returns
    -------

    sub_ids: list of str
             List containing the IDs of the desired number of subjects
    """

    sub_ids = [dir_ for dir_ in os.listdir(root_dir) if dir_.startswith('CC')]

    if n_sub is not None:
        sub_ids = sub_ids[:n_sub]

    return sub_ids


def load_camcan_recording(root_dir, subj, record_type, show_figs=False, 
                          results_dir=None):
    """Load a CAMCAN recording.
    """
    fname = get_camcam_recording_fname(root_dir, subj, record_type)[0]
    raw = read_raw_fif(fname, preload=True)

    mne.channels.fix_mag_coil_types(raw.info)  # Change coil types for Neuromag

    if show_figs or results_dir is not None:
        fig = raw.plot_psd(show=show_figs)
    if results_dir is not None:
        fig.savefig(os.path.join(results_dir, '1a_raw_psd.png'))

    return raw


def save_files(file_list, save_dir, prefix, suffix='', extension='fif'):
    """
    Save a file to disk, to keep track of intermediate steps of the processing

    Parameters
    ----------

    file_list: list of tuples
               First element of the tuple is the subject ID, and the second is the file to save

    save_dir: str or path object
              Absolute path to save the file to

    prefix: str
            String to append before the subject ID. Related to the processing step

    suffix: str, default ''
            String to append after the subject ID. Related to the type of file if any

    extension: str, default 'fif'
               File extension for the output files. Do not include the dot
    """

    for sub_id, data in file_list:
        filename = f"{prefix}-{sub_id}_{suffix}.{extension}"
        save_path = os.path.join(save_dir, f"{sub_id}/{filename}")
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(save_path)

        data.save(save_path, overwrite=True)


def run_maxfilter(sub_id, raw, cal_fname, ctc_fname, show_figs=False,
                  results_dir=None):
    """Run maxfilter on one subject"""
    raw = maxwell_filter(raw, origin='auto', calibration=cal_fname, 
                         cross_talk=ctc_fname, st_duration=10)

    if show_figs or results_dir is not None:
        fig = raw.plot_psd(show=show_figs)
    if results_dir is not None:
        sub_dir = os.path.join(results_dir, sub_id)  # Create a dir with the ID name
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
        fig.savefig(os.path.join(sub_dir, f'sss_{sub_id}_psd.png'))

    return sub_id, raw


def filter_data(sub_id, raw, l_freq=1, h_freq=40, power_line=50, show_figs=False,
                results_dir=None):
    """
    Applying bandpass filter.

    Evoked responses are usually below 30-40 Hz. E.g., Gramfort et al. 2014 
    (Front. in Neur.). We also add a low cutoff frequency highpass to perform 
    some detrending, which is specifically needed for ICA.

    Parameters
    ----------

    sub_id: string
            ID of the subject

    raw: mne.Raw object
         Data to filter

    l_freq: int
            Low frequency for the filter

    h_freq: int
            High frequency for the filter

    power_line: int
                Base amount for power-line based notch filtering. Depending
                on the source of the data, this can be 50 or 60 Hz

    Returns
    -------

    raw: mne.Raw object
         Raw data after being filtered
    """
    raw.notch_filter(freqs=np.arange(power_line, (power_line * 4) + 1, power_line))
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')

    if show_figs or results_dir is not None:
        fig = raw.plot_psd(show=show_figs)
    if results_dir is not None:
        sub_dir = os.path.join(results_dir, sub_id)  # Create a dir with the ID name
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
        fig.savefig(os.path.join(sub_dir, f'filtered_{sub_id}_psd.png'))

    return sub_id, raw


def run_ica_correction(sub_id, raw, method='picard', reject=None, decim=3,
                       random_state=42, show_figs=False, results_dir=None):
    """
    Fit and apply ICA to correct heartbeats and blinks

    Parameters
    ----------

    sub_id: str
            Subject ID

    raw: mne.Raw object
         Data to apply ICA to

    method: str, default 'picard'
            Method for the ICA algorithm. See the documentation on the MNE webpage for other
            options

    reject: None or dict, default None
            Reject values previous to the ICA fit and application

    decim: int, default 3
           ICA parameter. Check out the documentation for more info

    random_state: int, default 42
                  Seed for ICA random state

    Returns
    -------

    sub_id: str
            Subject ID, unchanged

    raw: mne.Raw object
         Our old friend, but now without blinks and hearbeats (hopefully)

    ica: mne.ICA object
         The ICA object with the configuration used to eliminate artifacts
    """
    # picks_meg = mne.pick_types(raw.info, meg=True, eog=True, stim=True,
    #                            exclude='bads')

    # The rank will be used as a reference for the ICA components
    rank = mne.compute_rank(raw)

    ica = ICA(n_components=rank['meg'], method=method, random_state=random_state)
    ica.fit(raw, decim=decim, reject=reject)  # {'mag': 5e-12, 'grad': 4000e-13}


    # Create eog/ecg epochs using our dedicated channels in the data, and then find data
    # segments containing artifacts
    eog_epochs = create_eog_epochs(raw, ch_name='EOG061', reject=None)
    eog_inds, _ = ica.find_bads_eog(eog_epochs)

    ecg_epochs = create_ecg_epochs(raw, reject=None)
    ecg_inds, _ = ica.find_bads_ecg(ecg_epochs)

    # Compute an average of our eog/ecg epochs
    eog_avg = eog_epochs.average()
    ecg_avg = ecg_epochs.average()

    if show_figs or results_dir is not None:
        fig_eog = ica.plot_overlay(eog_avg, exclude=eog_inds, show=show_figs)
        fig_ecg = ica.plot_overlay(ecg_avg, exclude=ecg_inds, show=show_figs)
    if results_dir is not None:
        sub_dir = os.path.join(results_dir, sub_id)  # Create a dir with the ID name
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)

        fig_eog.savefig(os.path.join(sub_dir, f'{sub_id}_eog_correction.png'))
        fig_ecg.savefig(os.path.join(sub_dir, f'{sub_id}_ecg_correction.png'))

    # Tell our friend ICA what to do, and apply to data
    ica.exclude.extend(eog_inds)
    ica.exclude.extend(ecg_inds)
    ica.apply(raw)

    return sub_id, raw, ica


def extract_epochs_camcan(raw, tmin=-0.2, tmax=0.5, event_id=None):
    """Extract epochs or sliding windows from CAMCAN recording.
    """
    if event_id is None:  # empty room, use sliding windows
        events = mne.make_fixed_length_events(
            raw, start=max(-tmin, 0), duration=tmax-tmin)
    else:
        events = mne.find_events(raw)

    epochs = mne.Epochs(
        raw, events, event_id, tmin, tmax, baseline=(None, 0), reject=None, 
        verbose=False, detrend=0, preload=True)

    return epochs


def run_autoreject(epochs, show_figs=False, results_dir=None):
    """Run autoreject.
    """
    from autoreject import AutoReject

    ar = AutoReject()
    epochs = ar.fit_transform(epochs)

    if show_figs or results_dir is not None:
        pass
        # ar_log = ar.get_reject_log(epochs_clean)
        # fig_log = ar_log.plot()
        # ar_log.plot_epochs()
        # Similar to bad_segments, but with entries 0, 1, and 2.
        #     0 : good data segment
        #     1 : bad data segment not interpolated
        #     2 : bad data segment interpolated
    if results_dir is not None:
        pass
        # fig_log.savefig(os.path.join(results_dir, '4a_bad_epochs.png'))

    return epochs


def compute_evoked(epochs, show_figs=False, results_dir=None):
    """Compute evoked response from CAMCAN data.
    """
    picks_meg = mne.pick_types(epochs.info)
    evokeds = epochs.average(picks=picks_meg)

    if show_figs or results_dir is not None:
        ts_args = dict(gfp=True, time_unit='s')
        topomap_args = dict(sensors=True, time_unit='s')
        for i, name in enumerate(epochs.event_id.keys()):
            evoked = epochs[name].average(picks=picks_meg)
            figs = evoked.plot_joint(title=name, times='peaks', ts_args=ts_args,
                                     topomap_args=topomap_args, show=show_figs)
            if results_dir is not None:
                for i, fig in enumerate(figs):
                    fig.savefig(os.path.join(
                        results_dir, '5_{}_{}.png'.format(name, i + 1)))

    return evokeds


def compute_noise_cov(epochs, evokeds=None, show_figs=False, results_dir=None):
    """Compute noise covariance.
    """
    noise_cov = mne.compute_covariance(
        epochs, tmax=0, method=['empirical', 'diagonal_fixed', 'shrunk'])

    if (show_figs or results_dir is not None) and evokeds is not None:
        fig_noise_covs = noise_cov.plot(epochs.info, show=show_figs)
        fig_epochs_cov = evokeds.plot_white(
            noise_cov, time_unit='s', show=show_figs)
    if results_dir is not None:
        for i, fig in enumerate(fig_noise_covs):
            fig.savefig(os.path.join(
                results_dir, '6a_noise_cov{}.png'.format(i + 1)))
        fig_epochs_cov.savefig(
            os.path.join(results_dir, '6b_epochs_noise_cov.png'))

    return noise_cov


def make_save_fnames(save_dir, subj, record_type):
    """Generate filenames for saving intermediate processing steps.

    Args:
        save_dir (str): directory in which to create a subject folder.
        subj (str): subject name (e.g., 'CC110033'). A subdirectory with this
            name will be created under `save_dir`.
        record_type (str): type of CAMCAN recording ('passive', 'task' or 'rest')

    Returns:
        (str): directory where the results for the specified subject and 
            record_type will be saved.
        (dict): dictionary containing the filenames to use for saving the 
            results of intermediate processing steps.
    """
    results_dir = os.path.join(save_dir, subj, record_type)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    save_fnames = dict()
    save_fnames['maxfilter'] = os.path.join(
        results_dir, '{}_maxfilter_raw_sss.fif'.format(record_type))
    save_fnames['ica_clean'] = os.path.join(
        results_dir, '{}_ica_raw.fif'.format(record_type))
    save_fnames['ica'] = os.path.join(results_dir, 'ica.fif')
    save_fnames['clean'] = os.path.join(
        results_dir, '{}_clean_epochs-epo.fif'.format(record_type))
    save_fnames['evoked'] = os.path.join(
        results_dir, '{}_evoked-ave.fif'.format(record_type))
    save_fnames['noise_cov'] = os.path.join(
        results_dir, '{}_noise_cov.fif'.format(record_type))
    save_fnames['noise_cov_empty'] = os.path.join(
        results_dir, '{}_noise_cov_empty.fif'.format(record_type))

    return results_dir, save_fnames


def run_camcan_pipeline(root_dir, subj, record_type, save_dir, 
                        empty_room_dir=None, save_figs=True, show_figs=False,
                        save_intermediate=True, next_step=None, apply_ica=True,
                        apply_autoreject=True):
    """Run CAMCAN pipeline.

    Process CAMCAN data from a single subject and task type and produce
    diagnostic graphs at each step.

    Args:
        root_dir (str): root directory where the CAMCAN fif files are saved. 
            Inside that root directory the data should follow the structure:

                /<subj1>/passive/passive_raw.fif
                        /task/task_raw.fif
                        /rest/rest_raw.fif
                /<subj2>/...

        subj (str): subject name (e.g., 'CC110033'). A subdirectory with this
            name will be created under `save_dir`.
        record_type (str): type of CAMCAN recording ('passive', 'task' or 'rest')
        save_dir (str): directory in which to create a subject folder.

    Keyword Args:
        empty_room_dir (str): root directory where the CAMCAN empty room fif
            files are. If None, do not compute the empty room covariance matrix.
        save_figs (bool): if True, save the figures as png. The output of this
            function is the directory where these figures will be saved.
        show_figs (bool): if True, show the figures.
        save_intermediate (bool): if True, save intermediate results at each 
            relevant step.
        next_step (str): processing step to start with. Can be None, 'maxfilter',
            'ica', 'clean', 'evoked', 'cov' or 'empty_room_cov'. If None or 
            'maxfilter', the processing will start at the beginning (from the
            raw file). Otherwise, the intermediate results from the previous 
            step will be loaded and the pipeline will continue from there.
        apply_ica (bool): if True, run ICA and perform ocular and heart artefact
            component rejection. If False, skip this step.
        apply_autoreject (bool): if True, reject artefacted epochs with 
            autoreject.

    Returns:
        (str): directory where the results for the specified subject and 
            record_type will be saved.
    """
    if record_type == 'passive':
        event_id = {'Auditory 300Hz': 6,  # See trigger_codes.txt
                    'Auditory 600Hz': 7,
                    'Auditory 1200Hz': 8,
                    'Visual Checkerboard': 9}
    elif record_type == 'task':
        raise NotImplementedError('Need to add information about tasks events!')
    elif record_type == 'rest':
        raise NotImplementedError('Need to extract sliding windows!')
    else:
        raise ValueError('record_type must be `passive`, `task` or `rest`; got '
                         '{}.'.format(record_type))

    results_dir, save_fnames = make_save_fnames(save_dir, subj, record_type)
    results_dir = results_dir if save_figs else None

    l_freq, h_freq = 1, 40

    if next_step is None or next_step == 'maxfilter':
        raw = load_camcan_recording(
            root_dir, subj, record_type, results_dir=results_dir)
        raw = run_maxfilter(raw, cal_fname, ctc_fname, show_figs=show_figs, 
                            results_dir=results_dir)
        if save_intermediate:
            raw.save(save_fnames['maxfilter'], overwrite=True)
        next_step = 'ica'
            
    if next_step == 'ica' and apply_ica:
        if 'raw' not in locals():
            raw = read_raw_fif(save_fnames['maxfilter'], preload=True)
        raw = run_bandpass_filter(
            raw, l_freq=l_freq, h_freq=h_freq, show_figs=show_figs, 
            results_dir=results_dir)
        raw, ica = run_ica_correction(
            raw, show_figs=show_figs, results_dir=results_dir)
        if save_intermediate:
            ica.save(save_fnames['ica'])
            raw.save(save_fnames['ica_clean'], overwrite=True)
        next_step = 'clean'

    # 4. Reject artifacted epochs with autoreject
    if next_step == 'clean' and apply_autoreject:
        if 'raw' not in locals():
            raw = read_raw_fif(save_fnames['ica_clean'], preload=True)
        epochs = extract_epochs_camcan(raw, tmin=-0.2, tmax=0.5, event_id=event_id)
        epochs = run_autoreject(
            epochs, show_figs=show_figs, results_dir=results_dir)
        if save_intermediate:
            epochs.save(save_fnames['clean'])
        next_step = 'evoked'

    # 5. Compute evoked responses
    if next_step == 'evoked':
        if 'epochs' not in locals():
            epochs = mne.read_epochs(save_fnames['clean'], preload=True)
        evokeds = compute_evoked(
            epochs, show_figs=show_figs, results_dir=results_dir) 
        if save_intermediate:
            evokeds.save(save_fnames['evoked'])
        next_step = 'cov'

    # 6. Compute the noise covariance from baseline segments
    if next_step == 'cov':
        if 'epochs' not in locals():
            epochs = mne.read_epochs(save_fnames['clean'], preload=True)
        noise_cov = compute_noise_cov(
            epochs, evokeds=evokeds, show_figs=show_figs, results_dir=results_dir)
        if save_intermediate:
            noise_cov.save(save_fnames['noise_cov'])
        next_step = 'empty_room_cov'
        
    # 7. Compute the noise covariance from the empty room recording
    if next_step == 'empty_room_cov' and empty_room_dir is not None:
        if 'ica' not in locals():
            ica = read_raw_fif(save_fnames['ica'], preload=True)
        # TODO: Remove same bad channels as above
        show_empty_room = False
        raw_empty = load_camcan_recording(
            empty_room_dir, subj, record_type, show_figs=show_empty_room, 
            results_dir=None)
        raw_empty = run_maxfilter(
            raw_empty, cal_fname, ctc_fname, show_figs=show_empty_room,
            results_dir=None)
        raw_empty = run_bandpass_filter(
            raw_empty, l_freq=l_freq, h_freq=h_freq, show_figs=show_empty_room, 
            results_dir=None)
        ica.apply(raw_empty)
        # TODO: make it possible to compute covariance without providing an
        #       evoked object!
        epochs_empty = extract_epochs_camcan(
            raw_empty, tmin=-0.2, tmax=0.5, event_id=None)
        epochs_empty = run_autoreject(
            epochs_empty, show_figs=show_empty_room, results_dir=None)
        noise_cov_empty = mne.compute_covariance(
            epochs_empty, tmax=0, method=['empirical', 'diagonal_fixed', 'shrunk'],
            show_figs=show_empty_room)

        if save_intermediate:
            noise_cov_empty.save(save_fnames['noise_cov_empty'])

    # 8. Apply co-registration
    # ...

    # 9. Prepare inverse solution computation
    # ...

    # 10. Compute dSPM solution
    # ...

    if show_figs:
        plt.show()

    return results_dir


def get_data_paths():
    """Get data paths so the analysis works on different computers.
    """
    hostname = socket.gethostname()

    if 'drago' in hostname:
        root_dir = '/storage/store/data/camcan/camcan47/cc700/meg/pipeline/release004/data/aamod_meg_get_fif_00001'
        # save_dir = '/storage/store/work/hjacobba/data/CAMCAN/results'
        save_dir = '/storage/inria/hjacobba/mne_data/camcan/results'
        sss_dir = '/storage/store/work/hjacobba/data/CAMCAN/sss_params'
        # coreg_files = '/storage/store/data/camcan-mne/freesurfer/CC110033'
    else:
        root_dir = '/home/jtorretr/neuroimaging/meg_data/camcan'
        save_dir = '/home/jtorretr/neuroimaging/meg_data/camcan/results'
        sss_dir = '/home/jtorretr/neuroimaging/meg_data/camcan/sss_params'
        
    cal_fname = os.path.join(sss_dir, 'sss_cal.dat')
    ctc_fname = os.path.join(sss_dir, 'ct_sparse.fif')

    return root_dir, save_dir, cal_fname, ctc_fname

