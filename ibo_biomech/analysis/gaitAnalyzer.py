'''
Basic gait analysis functions adapted from Marvin Zedler.
The original code was only modified to work with trialData objects rather than raw c3d files. 
All the functionality is currently decoupled from the rest of the package and can be used independently. 
Further refactoring and improvements can be made. 
'''

from scipy.signal import find_peaks
import numpy as np

class GaitAnalyzer:
    """Gait analysis functionality for biomechanics data.
    
    Provides methods for analyzing gait events such as foot contacts detected
    from force plate data and calculating stance times and step lengths from
    motion capture markers. Adapted from original code by Marvin Zedler to work
    with TrialData objects.
    """

    @staticmethod
    def get_plate_contacts(trialdata, bodyweight, marker_re, marker_li, threshold=20, threshold_multiplier=1.2, prominence_multiplier=0.8):
        """Detect force-peak contacts on synchronized, Z-up trial data.

        Force samples must share the analog rate and the marker time origin.
        Use complete contacts with one qualifying peak each and plates unloaded
        at the first sample. Foot assignment compares the two markers' heights.

        Args:
            trialdata: TrialData containing markers, forces and an analog rate.
            bodyweight: Body weight as a force in newtons, not mass in kilograms.
            marker_re: Right foot marker name.
            marker_li: Left foot marker name.
            threshold: Touchdown/toe-off force threshold in newtons. Defaults to 20.
            threshold_multiplier: Body-weight multiplier for peak height.
                Defaults to 1.2.
            prominence_multiplier: Body-weight multiplier for peak prominence.
                Defaults to 0.8.

        Returns:
            dict: Contacts sorted by force-peak sample index. ``plateNames``
            contains string mapping keys such as ``forceplate_0`` and ``feet``
            contains ``L``/``R``. ``TDa``/``TOa`` are force-array indices;
            ``TDv``/``TOv`` are rounded marker-array indices, not source frame
            numbers. All values are ``None`` when no contacts are found.
            The result does not populate ``trialdata.events``.
        """
        # Constants
        tdto_threshold = threshold # Newton
        contact_threshold = threshold_multiplier*bodyweight # for running and sprinting
        peakDist = trialdata.analog_rate/10 # at least 100ms apart
        peakProminence = prominence_multiplier*bodyweight

        plate_names = trialdata.forces.keys()
        analog_per_video = trialdata.analog_rate / trialdata.marker_rate
        markers = trialdata.markers

        contacts = []
        feet = []
        TD = []
        TO = []
        Times = []

        # loop over all force plates
        for i, plate_name in enumerate(plate_names):
            fz = trialdata.forces[plate_name].Fz

            # find peaks in Fz with required prominence
            peak_indices, properties = find_peaks(fz, height=contact_threshold, distance=peakDist, prominence=peakProminence, width=peakDist)
            peak_values = fz[peak_indices]
    
            for peak_idx, peak_val in zip(peak_indices, peak_values):
                # skip if below threshold or already standing on plate at start
                if peak_val < contact_threshold or fz[0] > tdto_threshold:
                    continue

                # determine TD and TO around this peak
                td_idx, to_idx = GaitAnalyzer.find_TDTO(fz, peak_idx, tdto_threshold)

                TD.append(td_idx)
                TO.append(to_idx)

                # convert analog index to video frame index
                max_idx_video = int(round(peak_idx / analog_per_video))

                Times.append(peak_idx)
                contacts.append(plate_name)  # 0-based; use i+1 if you want MATLAB-like plate numbering

                # decide which foot: compare marker heights 
                if markers[marker_li].z[max_idx_video] < markers[marker_re].z[max_idx_video]:
                    feet.append("L")
                else:
                    feet.append("R")

        # if no contacts detected
        if len(Times) == 0:
            return {
                "plateNames": None,
                "feet": None,
                "TDa": None,
                "TOa": None,
                "TDv": None,
                "TOv": None
            }

        # sort everything by time
        sort_order = np.argsort(Times)

        contacts_sorted = np.array([contacts[idx] for idx in sort_order])
        feet_sorted = np.array([feet[idx] for idx in sort_order])
        TD_sorted = np.array([TD[idx] for idx in sort_order])
        TO_sorted = np.array([TO[idx] for idx in sort_order])
        TDv = np.round(TD_sorted / analog_per_video).astype(int)
        TOv = np.round(TO_sorted / analog_per_video).astype(int)
        events = {
                "plateNames": contacts_sorted,
                "feet": feet_sorted,
                "TDa": TD_sorted,
                "TOa": TO_sorted,
                "TDv": TDv,
                "TOv": TOv,
                }
        return events
                

    @staticmethod
    def find_TDTO(Fz, max_idx, threshold):
        """
        Python version of nested findTDTO_kienbaum in MATLAB.
        Searches backward and forward from max_idx until Fz falls below 'threshold'.
        Indices are analog sample indices (0-based).
        """

        TD = np.nan
        TO = np.nan

        # search backwards for touchdown
        for i in range(max_idx, -1, -1):
            if Fz[i] < threshold:
                TD = i + 1  # first index where Fz >= threshold (approx.)
                break

        # search forwards for toe-off
        for i in range(max_idx, len(Fz)):
            if Fz[i] < threshold:
                TO = i - 1  # last index where Fz >= threshold
                break

        if np.isnan(TD):
            print("Detection of Touchdown failed")
        if np.isnan(TO):
            print("Detection of Toeoff failed")

        return TD, TO

    @staticmethod
    def get_stancetimes_steplengths(markerDict, events, aFrq):
        """Calculate stance times and step lengths from detected gait events.
        
        Requires markers named ``LTOE`` and ``RTOE`` and complete contact indices.
        Step displacement is along X between consecutive touchdown positions,
        in the marker length unit.
        
        Args:
            markerDict : dict
                Dictionary mapping marker names to marker data objects.
            events : dict
                Dictionary of gait events from get_plate_contacts() containing
                "feet", "TDa", "TOa", and "TDv" keys.
            aFrq : float
                Analog sample rate (Hz).
                
        Returns:
            tuple: Stance times in seconds and step displacements. The first
            step displacement is ``None``.
        """
        stanceTimes = []
        stepLengths = []
        stepLengths.append(None)
        for c in range(len(events["feet"])):
            stanceTimes.append((events["TOa"][c] - events["TDa"][c]) / aFrq)
            if c > 0:
                foot = events["feet"][c]+"TOE"
                previous_foot = events["feet"][c-1]+"TOE"
                TDv = events["TDv"][c]
                TDv_prev = events["TDv"][c-1]
                stepLengths.append(markerDict[foot].x[TDv] - markerDict[previous_foot].x[TDv_prev]) 
        return stanceTimes, stepLengths
