"""WAY-EEG-GAL data loader for hand pose regression."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import scipy.io as sio
from scipy import signal
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class TrialData:
    """Container for a single trial's data."""
    participant: int
    series: int
    trial_idx: int
    eeg: np.ndarray           # (N_samples, 32) at 500 Hz
    kin: np.ndarray           # (N_samples, 45) at 500 Hz
    emg: np.ndarray           # (M_samples, 5) at 4000 Hz
    eeg_t: np.ndarray         # (N_samples,) EEG time in seconds
    emg_t: np.ndarray         # (M_samples,) EMG time in seconds
    trial_start_time: float
    led_on: float
    led_off: float
    trial_end_time: float
    weight: int
    weight_id: str
    surf: int
    surf_id: str
    
    # Events from AllLifts (aligned by trial index)
    t_hand_start: Optional[float] = None
    t_lift_off: Optional[float] = None
    t_replace: Optional[float] = None
    t_both_released: Optional[float] = None
    t_hand_stop: Optional[float] = None
    t_peak_vel_reach: Optional[float] = None
    t_peak_vel_retract: Optional[float] = None
    grip_aperture_max: Optional[float] = None
    t_grip_aperture_max: Optional[float] = None
    dur_reach: Optional[float] = None
    dur_preload: Optional[float] = None
    dur_load_phase: Optional[float] = None
    dur_release: Optional[float] = None
    gf_max: Optional[float] = None
    lf_max: Optional[float] = None
    gf_hold: Optional[float] = None
    lf_hold: Optional[float] = None


@dataclass
class MEWindow:
    """Container for a Movement Execution window."""
    trial: TrialData
    start_idx: int          # Index in EEG samples
    end_idx: int            # Index in EEG samples
    start_time: float       # Time relative to trial start
    end_time: float         # Time relative to trial start
    eeg: np.ndarray         # (window_samples, 32)
    kin: np.ndarray         # (window_samples, 45)
    emg: np.ndarray         # (window_samples, 5) - resampled to 500 Hz
    
    # Hold period within window (in window-relative time)
    hold_start_idx: Optional[int] = None
    hold_end_idx: Optional[int] = None
    hold_start_time: Optional[float] = None
    hold_end_time: Optional[float] = None


@dataclass
class Targets:
    """Regression targets for a trial."""
    trial: TrialData
    window: MEWindow
    
    # Position targets (3D) - relative/displacement
    wrist_pos_rest: np.ndarray          # (3,) - P4 at rest (tHandStart or LEDOn)
    wrist_pos_hold: np.ndarray          # (3,) - P4 at hold (tLiftOff to tReplace)
    wrist_pos_lift_off: np.ndarray      # (3,) - P4 at lift-off
    wrist_pos_replace: np.ndarray       # (3,) - P4 at replace
    wrist_pos_disp: np.ndarray          # (3,) - displacement = hold - rest
    
    # Velocity targets (3D)
    wrist_vel_mean: np.ndarray          # (3,) - Mean velocity during reach (rest to lift-off)
    wrist_vel_peak: np.ndarray          # (3,) - Peak velocity during reach
    wrist_vel_at_lift_off: np.ndarray   # (3,) - Velocity at lift-off
    
    # Orientation targets (3D)
    wrist_orient_rest: np.ndarray       # (3,) - Ae4, Ar4, Az4 at rest
    wrist_orient_hold: np.ndarray       # (3,) - Ae4, Ar4, Az4 at hold
    wrist_orient_disp: np.ndarray       # (3,) - Orientation change
    
    # Grip targets (1D)
    grip_aperture_rest: float           # Distance P2-P3 at rest
    grip_aperture_hold: float           # Distance P2-P3 at hold
    grip_aperture_disp: float           # Grip aperture change
    grip_force_rest: float              # GF at rest
    grip_force_hold: float              # GF at hold
    grip_force_disp: float              # GF change
    
    # Metadata
    reach_duration: float               # tLiftOff - tHandStart (or LEDOn)
    hold_duration: float                # tReplace - tLiftOff


class WAYGALLoader:
    """Loader for WAY-EEG-GAL dataset."""
    
    # Channel names from the dataset
    EEG_CHANNELS = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
        'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
        'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10'
    ]
    
    KIN_COLUMNS = [
        'Ae1', 'Ae2', 'Ae3', 'Ae4',  # Angle e sensors 1-4
        'Ar1', 'Ar2', 'Ar3', 'Ar4',  # Angle r sensors 1-4
        'Az1', 'Az2', 'Az3', 'Az4',  # Angle z sensors 1-4
        'FX1', 'FX2',                # Force X plates 1-2
        'FY1', 'FY2',                # Force Y plates 1-2
        'FZ1', 'FZ2',                # Force Z plates 1-2
        'Px1', 'Px2', 'Px3', 'Px4',  # Position X sensors 1-4
        'Py1', 'Py2', 'Py3', 'Py4',  # Position Y sensors 1-4
        'Pz1', 'Pz2', 'Pz3', 'Pz4',  # Position Z sensors 1-4
        'TX1', 'TX2',                # Torque X plates 1-2
        'TY1', 'TY2',                # Torque Y plates 1-2
        'TZ1', 'TZ2',                # Torque Z plates 1-2
        'IndLF', 'ThuLF', 'LF',      # Load forces
        'IndGF', 'ThuGF', 'GF',      # Grip forces
        'IndRatio', 'ThuRatio', 'GFLFRatio'  # Ratios
    ]
    
    EMG_CHANNELS = [
        'Anterior Deltoid', 'Brachoradial', 'Flexor Digitorum',
        'Common Extensor Digitorum', 'First Dorsal Interosseus'
    ]
    
    def __init__(self, data_root: Path):
        """
        Initialize loader.
        
        Parameters
        ----------
        data_root : Path
            Root directory containing P1, P2, ... P12 subdirectories.
        """
        self.data_root = Path(data_root)
        self.participants = sorted([d.name for d in self.data_root.iterdir() 
                                    if d.is_dir() and d.name.startswith('P')])
        self._all_lifts_cache = {}
    
    def load_participant_all_lifts(self, participant: str) -> np.ndarray:
        """Load AllLifts array for a participant. Returns (n_trials, 43) float64 array."""
        if participant in self._all_lifts_cache:
            return self._all_lifts_cache[participant]
        
        all_lifts_path = self.data_root / participant / f"{participant}_AllLifts.mat"
        if not all_lifts_path.exists():
            raise FileNotFoundError(f"AllLifts not found: {all_lifts_path}")
        
        mat = sio.loadmat(str(all_lifts_path), squeeze_me=True, struct_as_record=False)
        P = mat['P']
        
        # AllLifts is a (n_trials, 43) float64 array
        all_lifts = P.AllLifts
        
        self._all_lifts_cache[participant] = all_lifts
        return all_lifts
    
    def load_participant_series(self, participant: str, series: int) -> List[TrialData]:
        """Load all trials from one series of a participant."""
        ws_path = self.data_root / participant / f"WS_{participant}_S{series}.mat"
        if not ws_path.exists():
            raise FileNotFoundError(f"Series file not found: {ws_path}")
        
        mat = sio.loadmat(str(ws_path), squeeze_me=True, struct_as_record=False)
        ws_obj = mat['ws']
        
        # Get events from AllLifts
        all_lifts_array = self.load_participant_all_lifts(participant)
        
        trials = []
        win = ws_obj.win
        for trial_idx, trial_data in enumerate(win):
            # Extract trial events from AllLifts
            events = self._get_trial_events(participant, series, trial_idx, all_lifts_array)
            
            trial = TrialData(
                participant=int(participant[1:]),
                series=series,
                trial_idx=trial_idx,
                eeg=trial_data.eeg,
                kin=trial_data.kin,
                emg=trial_data.emg,
                eeg_t=trial_data.eeg_t,
                emg_t=trial_data.emg_t,
                trial_start_time=float(trial_data.trial_start_time),
                led_on=float(trial_data.LEDon),
                led_off=float(trial_data.LEDoff),
                trial_end_time=float(trial_data.trial_end_time),
                weight=int(trial_data.weight),
                weight_id=str(trial_data.weight_id),
                surf=int(trial_data.surf),
                surf_id=str(trial_data.surf_id),
                t_hand_start=events.get('t_hand_start'),
                t_lift_off=events.get('t_lift_off'),
                t_replace=events.get('t_replace'),
                t_both_released=events.get('t_both_released'),
                t_hand_stop=events.get('t_hand_stop'),
                t_peak_vel_reach=events.get('t_peak_vel_reach'),
                t_peak_vel_retract=events.get('t_peak_vel_retract'),
                grip_aperture_max=events.get('grip_aperture_max'),
                t_grip_aperture_max=events.get('t_grip_aperture_max'),
                dur_reach=events.get('dur_reach'),
                dur_preload=events.get('dur_preload'),
                dur_load_phase=events.get('dur_load_phase'),
                dur_release=events.get('dur_release'),
                gf_max=events.get('gf_max'),
                lf_max=events.get('lf_max'),
                gf_hold=events.get('gf_hold'),
                lf_hold=events.get('lf_hold'),
            )
            trials.append(trial)
        
        return trials
    
    def _get_trial_events(self, participant: str, series: int, trial_idx: int, 
                          all_lifts_data: np.ndarray) -> Dict[str, Optional[float]]:
        """Get events for a specific trial from AllLifts array."""
        part_num = int(participant[1:])
        events = {}
        
        # Column indices based on ColNames
        col_idx = {
            'Part': 0, 'Run': 1, 'Lift': 2,
            'LEDOn': 8, 'LEDOff': 9,
            'tLiftOff': 18, 'tReplace': 19, 'tBothReleased': 22,
            'tHandStart': 33, 'tHandStop': 34,
            'tPeakVelHandReach': 35, 'tPeakVelHandRetract': 36,
            'GripAparture_Max': 37, 'tGripAparture_Max': 38,
            'Dur_Reach': 39, 'Dur_Preload': 40, 'Dur_LoadPhase': 41, 'Dur_Release': 42,
            'GF_Max': 23, 'LF_Max': 24, 'GF_Hold': 31, 'LF_Hold': 32,
        }
        
        # Map AllLifts column names to TrialData field names
        event_to_field = {
            'tHandStart': 't_hand_start',
            'tLiftOff': 't_lift_off',
            'tReplace': 't_replace',
            'tBothReleased': 't_both_released',
            'tHandStop': 't_hand_stop',
            'tPeakVelHandReach': 't_peak_vel_reach',
            'tPeakVelHandRetract': 't_peak_vel_retract',
            'GripAparture_Max': 'grip_aperture_max',
            'tGripAparture_Max': 't_grip_aperture_max',
            'Dur_Reach': 'dur_reach',
            'Dur_Preload': 'dur_preload',
            'Dur_LoadPhase': 'dur_load_phase',
            'Dur_Release': 'dur_release',
            'GF_Max': 'gf_max',
            'LF_Max': 'lf_max',
            'GF_Hold': 'gf_hold',
            'LF_Hold': 'lf_hold',
        }
        
        # Match by participant, run (series), and lift (trial)
        part_col = all_lifts_data[:, col_idx['Part']]
        run_col = all_lifts_data[:, col_idx['Run']]
        lift_col = all_lifts_data[:, col_idx['Lift']]
        
        mask = (part_col == part_num) & (run_col == series) & (lift_col == trial_idx + 1)
        matching_indices = np.where(mask)[0]
        
        if len(matching_indices) == 1:
            idx = matching_indices[0]
            # Extract event columns
            for event_name, col in col_idx.items():
                if event_name not in event_to_field:
                    continue
                
                field_name = event_to_field[event_name]
                val = all_lifts_data[idx, col]
                # Convert to float, handling NaN
                if isinstance(val, (np.number, int, float)) and not np.isnan(val):
                    events[field_name] = float(val)
                else:
                    events[field_name] = None
        
        return events
    
    def load_all_participants(self) -> Dict[str, Dict[int, List[TrialData]]]:
        """Load all data for all participants and series."""
        all_data = {}
        for participant in self.participants:
            all_data[participant] = {}
            for series in range(1, 10):
                try:
                    trials = self.load_participant_series(participant, series)
                    if trials:
                        all_data[participant][series] = trials
                        print(f"  {participant} S{series}: {len(trials)} trials")
                except Exception as e:
                    print(f"  {participant} S{series}: ERROR - {e}")
        return all_data


def resample_emg_to_eeg(emg: np.ndarray, emg_t: np.ndarray, eeg_t: np.ndarray) -> np.ndarray:
    """Resample EMG from 4000 Hz to EEG 500 Hz using linear interpolation."""
    if len(emg_t) == 0 or len(eeg_t) == 0:
        return np.zeros((len(eeg_t), emg.shape[1]))
    
    # Interpolate each EMG channel
    emg_resampled = np.zeros((len(eeg_t), emg.shape[1]))
    for ch in range(emg.shape[1]):
        interp = interp1d(emg_t, emg[:, ch], kind='linear', 
                         bounds_error=False, fill_value='extrapolate')
        emg_resampled[:, ch] = interp(eeg_t)
    
    return emg_resampled


def savgol_derivative(pos: np.ndarray, fs: float, window: int = 11, poly: int = 3) -> np.ndarray:
    """Compute velocity using Savitzky-Golay filter."""
    # pos: (N, 3) - positions
    # Returns velocity: (N, 3)
    vel = np.zeros_like(pos)
    for i in range(3):
        vel[:, i] = signal.savgol_filter(pos[:, i], window, poly, deriv=1, delta=1/fs)
    return vel


def extract_me_windows(
    trials: List[TrialData],
    fs: float = 500.0,
    pre_handstart: float = 0.2,      # 200 ms before hand start
    post_replace: float = 0.3,       # 300 ms after replace
    min_duration: float = 0.5,       # Minimum window duration
    max_duration: float = 4.0,       # Maximum window duration
) -> List[MEWindow]:
    """
    Extract Movement Execution windows from trials.
    
    Window: [tHandStart - pre_handstart, tReplace + post_replace] or [LEDOn, tReplace + post_replace]
    This covers the full reach-to-hold period.
    """
    windows = []
    
    for trial in trials:
        # Determine window boundaries
        if trial.t_hand_start is not None and trial.t_replace is not None:
            # Use actual events: from hand start to replace
            start_time = trial.t_hand_start - pre_handstart
            end_time = trial.t_replace + post_replace
        elif trial.t_hand_start is not None and trial.t_lift_off is not None:
            # Fallback: hand start to lift-off + buffer
            start_time = trial.t_hand_start - pre_handstart
            end_time = trial.t_lift_off + 0.5
        elif trial.t_hand_start is not None:
            start_time = trial.t_hand_start - pre_handstart
            end_time = start_time + max_duration
        elif trial.t_replace is not None:
            end_time = trial.t_replace + post_replace
            start_time = end_time - max_duration
        else:
            # Fall back to LEDOn-based window
            start_time = 0.0  # LEDOn
            end_time = 3.0
        
        # Clamp to trial duration
        start_time = max(start_time, 0)
        end_time = min(end_time, trial.eeg_t[-1] - trial.eeg_t[0])
        
        if end_time - start_time < min_duration:
            continue
        
        # Convert to sample indices
        trial_time = trial.eeg_t - trial.eeg_t[0]
        start_idx = int(np.round(start_time * fs))
        end_idx = int(np.round(end_time * fs))
        
        # Extract EEG window
        eeg_win = trial.eeg[start_idx:end_idx]
        
        # Extract kinematics window (already at 500 Hz)
        kin_win = trial.kin[start_idx:end_idx]
        
        # Resample EMG to EEG timebase
        emg_resampled = resample_emg_to_eeg(
            trial.emg, trial.emg_t, trial.eeg_t[start_idx:end_idx]
        )
        
        # Determine hold period within window
        hold_start_idx = hold_end_idx = None
        hold_start_time = hold_end_time = None
        
        if trial.t_lift_off is not None and trial.t_replace is not None:
            # Hold period: tLiftOff to tReplace
            hold_start_time = trial.t_lift_off - start_time
            hold_end_time = trial.t_replace - start_time
            hold_start_idx = int(np.round(hold_start_time * fs))
            hold_end_idx = int(np.round(hold_end_time * fs))
        elif trial.t_lift_off is not None:
            # Hold starts at lift-off, ends at window end
            hold_start_time = trial.t_lift_off - start_time
            hold_start_idx = int(np.round(hold_start_time * fs))
            hold_end_idx = kin_win.shape[0]
            hold_end_time = hold_end_idx / fs
        
        window = MEWindow(
            trial=trial,
            start_idx=start_idx,
            end_idx=end_idx,
            start_time=start_time,
            end_time=end_time,
            eeg=eeg_win,
            kin=kin_win,
            emg=emg_resampled,
            hold_start_idx=hold_start_idx,
            hold_end_idx=hold_end_idx,
            hold_start_time=hold_start_time,
            hold_end_time=hold_end_time,
        )
        windows.append(window)
    
    return windows


def extract_targets(
    windows: List[MEWindow],
    kin_columns: List[str] = None
) -> List[Targets]:
    """
    Extract regression targets from ME windows.
    
    Primary targets (for LOSO regression):
    - wrist_pos_disp: Displacement from rest to hold (P4_hold - P4_rest)
    - wrist_vel_mean: Mean velocity during reach (rest to lift-off)
    - wrist_vel_peak: Peak velocity during reach
    - wrist_vel_at_lift_off: Velocity at lift-off
    - wrist_orient_disp: Orientation change from rest to hold
    - grip_aperture_disp: Grip aperture change (hold - rest)
    - grip_force_disp: GF change (hold - rest)
    - wrist_orient_hold: Absolute orientation at hold
    - grip_aperture_hold: Absolute grip at hold
    - grip_force_hold: Absolute GF at hold
    
    Secondary targets (for reference):
    - Absolute positions at rest, lift-off, hold, replace
    """
    if kin_columns is None:
        kin_columns = WAYGALLoader.KIN_COLUMNS
    
    # Column indices
    col_idx = {name: i for i, name in enumerate(kin_columns)}
    
    # P4 (wrist) position indices
    px4_idx = col_idx.get('Px4')
    py4_idx = col_idx.get('Py4')
    pz4_idx = col_idx.get('Pz4')
    
    # P2 (index) and P3 (thumb) position indices
    px2_idx = col_idx.get('Px2')
    py2_idx = col_idx.get('Py2')
    pz2_idx = col_idx.get('Pz2')
    px3_idx = col_idx.get('Px3')
    py3_idx = col_idx.get('Py3')
    pz3_idx = col_idx.get('Pz3')
    
    # Wrist orientation indices
    ae4_idx = col_idx.get('Ae4')
    ar4_idx = col_idx.get('Ar4')
    az4_idx = col_idx.get('Az4')
    
    # Grip force index
    gf_idx = col_idx.get('GF')
    
    targets_list = []
    
    for window in windows:
        kin = window.kin  # (N_samples, 45)
        trial = window.trial
        
        # Wrist position P4
        if all(v is not None for v in [px4_idx, py4_idx, pz4_idx]):
            wrist_pos = kin[:, [px4_idx, py4_idx, pz4_idx]]  # (N, 3)
        else:
            wrist_pos = np.zeros((len(kin), 3))
        
        # Wrist velocity (Savitzky-Golay derivative)
        wrist_vel = savgol_derivative(wrist_pos, fs=500.0)
        
        # Wrist orientation
        if all(v is not None for v in [ae4_idx, ar4_idx, az4_idx]):
            wrist_orient = kin[:, [ae4_idx, ar4_idx, az4_idx]]
        else:
            wrist_orient = np.zeros((len(kin), 3))
        
        # Grip aperture (distance between P2 and P3)
        if all(v is not None for v in [px2_idx, py2_idx, pz2_idx, px3_idx, py3_idx, pz3_idx]):
            p2 = kin[:, [px2_idx, py2_idx, pz2_idx]]
            p3 = kin[:, [px3_idx, py3_idx, pz3_idx]]
            grip_aperture = np.linalg.norm(p2 - p3, axis=1)
        else:
            grip_aperture = np.zeros(len(kin))
        
        # Grip force
        if gf_idx is not None:
            grip_force = kin[:, gf_idx]
        else:
            grip_force = np.zeros(len(kin))
        
        # Time array for window
        time_win = np.arange(len(kin)) / 500.0
        
        # --- Define key timepoints within window ---
        # Rest: tHandStart (or window start if not available)
        rest_time = 0.0  # window.start_time is relative to trial, but we use window-relative time
        if trial.t_hand_start is not None:
            rest_time = trial.t_hand_start - window.start_time
        rest_idx = int(np.round(rest_time * 500.0))
        rest_idx = max(0, min(rest_idx, len(kin) - 1))
        
        # Lift-off
        liftoff_idx = None
        liftoff_time = None
        if trial.t_lift_off is not None:
            liftoff_time = trial.t_lift_off - window.start_time
            liftoff_idx = int(np.round(liftoff_time * 500.0))
            liftoff_idx = max(0, min(liftoff_idx, len(kin) - 1))
        
        # Hold period
        hold_start_idx = window.hold_start_idx if window.hold_start_idx is not None else rest_idx
        hold_end_idx = window.hold_end_idx if window.hold_end_idx is not None else len(kin) - 1
        hold_start_idx = max(0, min(hold_start_idx, len(kin) - 1))
        hold_end_idx = max(0, min(hold_end_idx, len(kin) - 1))
        
        # Replace
        replace_idx = None
        if trial.t_replace is not None:
            replace_time = trial.t_replace - window.start_time
            replace_idx = int(np.round(replace_time * 500.0))
            replace_idx = max(0, min(replace_idx, len(kin) - 1))
        
        # --- Extract position targets ---
        if all(v is not None for v in [px4_idx, py4_idx, pz4_idx]):
            # Rest position (at tHandStart)
            wrist_pos_rest = wrist_pos[rest_idx]
            
            # Lift-off position
            wrist_pos_lift_off = wrist_pos[liftoff_idx] if liftoff_idx is not None else wrist_pos[-1]
            
            # Hold position (mean during hold)
            if hold_end_idx > hold_start_idx:
                wrist_pos_hold = np.mean(wrist_pos[hold_start_idx:hold_end_idx], axis=0)
            else:
                wrist_pos_hold = wrist_pos[liftoff_idx] if liftoff_idx is not None else wrist_pos[-1]
            
            # Replace position
            wrist_pos_replace = wrist_pos[replace_idx] if replace_idx is not None else wrist_pos[-1]
            
            # Displacement (primary target for LOSO)
            wrist_pos_disp = wrist_pos_hold - wrist_pos_rest
        else:
            wrist_pos_rest = np.zeros(3)
            wrist_pos_lift_off = np.zeros(3)
            wrist_pos_hold = np.zeros(3)
            wrist_pos_replace = np.zeros(3)
            wrist_pos_disp = np.zeros(3)
        
        # --- Extract velocity targets ---
        # Reach period: rest to lift-off
        if liftoff_idx is not None and liftoff_idx > rest_idx:
            reach_vel = wrist_vel[rest_idx:liftoff_idx]
            wrist_vel_mean = np.mean(reach_vel, axis=0) if len(reach_vel) > 0 else np.zeros(3)
            wrist_vel_peak = np.max(np.abs(reach_vel), axis=0) * np.sign(reach_vel[np.argmax(np.abs(reach_vel), axis=0), np.arange(3)]) if len(reach_vel) > 0 else np.zeros(3)
            wrist_vel_at_lift_off = wrist_vel[liftoff_idx]
        else:
            wrist_vel_mean = np.mean(wrist_vel[rest_idx:hold_start_idx], axis=0) if hold_start_idx > rest_idx else np.zeros(3)
            wrist_vel_peak = np.zeros(3)
            wrist_vel_at_lift_off = np.zeros(3)
        
        # --- Extract orientation targets ---
        if all(v is not None for v in [ae4_idx, ar4_idx, az4_idx]):
            wrist_orient_rest = wrist_orient[rest_idx]
            wrist_orient_hold = wrist_orient[hold_start_idx] if hold_start_idx < len(wrist_orient) else wrist_orient[-1]
            wrist_orient_disp = wrist_orient_hold - wrist_orient_rest
        else:
            wrist_orient_rest = np.zeros(3)
            wrist_orient_hold = np.zeros(3)
            wrist_orient_disp = np.zeros(3)
        
        # --- Extract grip targets ---
        grip_aperture_rest = grip_aperture[rest_idx] if rest_idx < len(grip_aperture) else 0
        grip_aperture_hold = grip_aperture[hold_start_idx] if hold_start_idx < len(grip_aperture) else 0
        grip_aperture_disp = grip_aperture_hold - grip_aperture_rest
        
        grip_force_rest = grip_force[rest_idx] if rest_idx < len(grip_force) else 0
        grip_force_hold = grip_force[hold_start_idx] if hold_start_idx < len(grip_force) else 0
        grip_force_disp = grip_force_hold - grip_force_rest
        
        # --- Metadata ---
        reach_duration = liftoff_time - rest_time if liftoff_time is not None else 0
        hold_duration = (hold_end_idx - hold_start_idx) / 500.0 if hold_end_idx > hold_start_idx else 0
        
        targets = Targets(
            trial=trial,
            window=window,
            
            # Position
            wrist_pos_rest=wrist_pos_rest,
            wrist_pos_hold=wrist_pos_hold,
            wrist_pos_lift_off=wrist_pos_lift_off,
            wrist_pos_replace=wrist_pos_replace,
            wrist_pos_disp=wrist_pos_disp,
            
            # Velocity
            wrist_vel_mean=wrist_vel_mean,
            wrist_vel_peak=wrist_vel_peak,
            wrist_vel_at_lift_off=wrist_vel_at_lift_off,
            
            # Orientation
            wrist_orient_rest=wrist_orient_rest,
            wrist_orient_hold=wrist_orient_hold,
            wrist_orient_disp=wrist_orient_disp,
            
            # Grip
            grip_aperture_rest=grip_aperture_rest,
            grip_aperture_hold=grip_aperture_hold,
            grip_aperture_disp=grip_aperture_disp,
            grip_force_rest=grip_force_rest,
            grip_force_hold=grip_force_hold,
            grip_force_disp=grip_force_disp,
            
            # Metadata
            reach_duration=reach_duration,
            hold_duration=hold_duration,
        )
        
        targets_list.append(targets)
    
    return targets_list


def save_targets(targets: List[Targets], output_path: Path):
    """Save targets to NPZ file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract arrays for saving
    n = len(targets)
    data = {
        'n_trials': n,
        'participant': np.array([t.trial.participant for t in targets]),
        'series': np.array([t.trial.series for t in targets]),
        'trial_idx': np.array([t.trial.trial_idx for t in targets]),
        
        # Primary targets (for LOSO)
        'wrist_pos_disp': np.stack([t.wrist_pos_disp for t in targets]),
        'wrist_vel_mean': np.stack([t.wrist_vel_mean for t in targets]),
        'wrist_vel_peak': np.stack([t.wrist_vel_peak for t in targets]),
        'wrist_vel_at_lift_off': np.stack([t.wrist_vel_at_lift_off for t in targets]),
        'wrist_orient_disp': np.stack([t.wrist_orient_disp for t in targets]),
        'grip_aperture_disp': np.array([t.grip_aperture_disp for t in targets]),
        'grip_force_disp': np.array([t.grip_force_disp for t in targets]),
        
        # Secondary targets (absolute, for reference)
        'wrist_pos_rest': np.stack([t.wrist_pos_rest for t in targets]),
        'wrist_pos_hold': np.stack([t.wrist_pos_hold for t in targets]),
        'wrist_pos_lift_off': np.stack([t.wrist_pos_lift_off for t in targets]),
        'wrist_pos_replace': np.stack([t.wrist_pos_replace for t in targets]),
        'wrist_vel_mean_abs': np.stack([t.wrist_vel_mean for t in targets]),
        'wrist_vel_peak': np.stack([t.wrist_vel_peak for t in targets]),
        'wrist_vel_at_lift_off': np.stack([t.wrist_vel_at_lift_off for t in targets]),
        'wrist_orient_rest': np.stack([t.wrist_orient_rest for t in targets]),
        'wrist_orient_hold': np.stack([t.wrist_orient_hold for t in targets]),
        'wrist_orient_disp': np.stack([t.wrist_orient_disp for t in targets]),
        'grip_aperture_rest': np.array([t.grip_aperture_rest for t in targets]),
        'grip_aperture_hold': np.array([t.grip_aperture_hold for t in targets]),
        'grip_aperture_disp': np.array([t.grip_aperture_disp for t in targets]),
        'grip_force_rest': np.array([t.grip_force_rest for t in targets]),
        'grip_force_hold': np.array([t.grip_force_hold for t in targets]),
        'grip_force_disp': np.array([t.grip_force_disp for t in targets]),
        
        # Metadata
        'reach_duration': np.array([t.reach_duration for t in targets]),
        'hold_duration': np.array([t.hold_duration for t in targets]),
    }
    
    np.savez_compressed(output_path, **data)
    print(f"Saved {n} targets to {output_path}")


def load_targets(npz_path: Path) -> Dict[str, np.ndarray]:
    """Load targets from NPZ file."""
    data = np.load(npz_path)
    return dict(data)