from pathlib import Path

import h5py
import numpy as np
import pytest
from ezc3d import c3d

from ibo_biomech import C3DHandler, Event, FileConverter, H5Handler, Subject, TrialData


@pytest.fixture
def event_c3d(synthetic_c3d):
    raw = c3d(str(synthetic_c3d))
    raw['header']['points']['first_frame'] = 6100
    raw.add_event([1, 1.02], label='Foot Strike', context='Left')
    raw.add_event([1, 1.08], label='Foot Strike', context='Right',
                  description='contact', subject='P1', icon_id=1, generic_flag=1)
    raw.add_event([1, 1.18], label='Foot Off', context='Left')
    raw.write(str(synthetic_c3d))
    return synthetic_c3d


def assert_events_equal(actual, expected):
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        assert left.time == pytest.approx(right.time, abs=1e-6)
        for field in ('name', 'frame', 'description'):
            assert getattr(left, field) == getattr(right, field)


def test_trial_events_keep_duplicates_and_crop_source_frames():
    trial = TrialData()
    left = Event('strike', np.int64(102), np.float64(1.02))
    right = Event('strike', 108, 1.08)
    trial.add_event(left)
    trial.add_event(right)
    assert trial.get_events('strike') == [left, right]
    assert trial.get_events('missing') == []
    assert trial.get_events() == [left, right]
    trial.crop('events', 105, 109)
    assert trial.events == [right]
    assert TrialData().events == []
    assert type(left.frame) is int and type(left.time) is float
    with pytest.raises(ValueError):
        trial.crop_events(10, 10)


@pytest.mark.parametrize('field,value', [('name', 1), ('frame', 1.5), ('frame', True),
    ('frame', -1), ('time', np.nan), ('time', np.inf), ('time', -1), ('time', '1')])
def test_invalid_events_rejected(field, value):
    kwargs = dict(name='event', frame=1, time=.01)
    kwargs[field] = value
    with pytest.raises(ValueError):
        Event(**kwargs)


def test_c3d_events_and_processed_edits(event_c3d, tmp_path):
    handler = C3DHandler(str(event_c3d))
    trial = handler.load_data()
    assert handler.events is trial.events
    assert [event.frame for event in trial.events] == [6102, 6108, 6118]
    assert [event.description for event in trial.get_events('Foot Strike')] == ['', 'contact']
    original = list(trial.events)
    trial.events.pop(0)
    trial.events[0] = Event('edited', 6109, 61.09)
    handler.add_event(Event('added', 6110, 61.1))
    output = tmp_path / 'edited.c3d'
    handler.write_c3d(str(output))
    assert_events_equal(C3DHandler(str(output)).load_data().events, trial.events)
    raw_output = tmp_path / 'raw.c3d'
    handler.write_raw_c3d(str(raw_output))
    assert_events_equal(C3DHandler(str(raw_output)).load_data().events, original)
    trial.events.clear()
    handler.write_c3d(str(output))
    assert C3DHandler(str(output)).load_data().events == []


def test_c3d_slice_crops_shared_event_list(event_c3d, tmp_path):
    handler = C3DHandler(str(event_c3d))
    trial = handler.load_data()
    handler.slice_c3d(5, 14)
    assert handler.events is trial.events
    assert [event.frame for event in trial.events] == [6108]
    output = tmp_path / 'cropped.c3d'
    handler.write_c3d(str(output))
    assert_events_equal(C3DHandler(str(output)).load_data().events, trial.events)


def test_c3d_crop_boundaries_use_frames_despite_float32_times(event_c3d, tmp_path):
    handler = C3DHandler(str(event_c3d))
    trial = handler.load_data()
    trial.events[:] = [Event('start', 6105, float(np.float32(61.05))),
                       Event('last', 6114, float(np.float32(61.14))),
                       Event('end', 6115, float(np.float32(61.15)))]
    handler.slice_c3d(5, 14)
    assert [event.name for event in trial.events] == ['start', 'last']
    output = tmp_path / 'boundaries.c3d'
    handler.write_c3d(str(output))
    assert_events_equal(C3DHandler(str(output)).load_data().events, trial.events)


def test_header_events_are_offset_to_marker_clock():
    handler = C3DHandler()
    handler.c3d_data = {
        'parameters': {'POINT': {'RATE': {'value': [100.]}}},
        'header': {'points': {'first_frame': 500},
                   'events': {'events_label': ['RHS', ''], 'events_time': [.08, 0.]}}
    }
    handler._parse_events()
    assert handler.events == [Event('RHS', 508, 5.08)]


def test_c3d_malformed_event_count_is_rejected(event_c3d):
    handler = C3DHandler(str(event_c3d))
    handler.load_data()
    handler.c3d_data['parameters']['EVENT']['USED']['value'] = [10]
    with pytest.raises(ValueError, match='matching event counts'):
        handler._parse_events()


def test_c3d_inconsistent_frame_rejected_atomically(event_c3d, tmp_path):
    handler = C3DHandler(str(event_c3d))
    trial = handler.load_data()
    trial.events[0].frame = 0
    output = tmp_path / 'keep.c3d'
    output.write_bytes(b'keep')
    with pytest.raises(ValueError, match='frame must match time'):
        handler.write_c3d(str(output))
    assert output.read_bytes() == b'keep'


def test_nonzero_origin_distinguishes_array_index_from_source_frame(event_c3d, tmp_path):
    handler = C3DHandler(str(event_c3d))
    trial = handler.load_data()
    marker = next(iter(trial.markers.values()))
    event = Event('from marker', frame=8, time=float(marker.time[8]))
    trial.events[:] = [event]
    with pytest.raises(ValueError, match='expected source frame 6108'):
        handler.write_c3d(str(tmp_path / 'bad.c3d'))
    event.frame = marker.first_frame + 8
    output = tmp_path / 'correct.c3d'
    handler.write_c3d(str(output))
    assert_events_equal(C3DHandler(str(output)).load_data().events, trial.events)


def test_c3d_rejects_silently_renumbering_frames_after_clock_shift(event_c3d, tmp_path):
    handler = C3DHandler(str(event_c3d))
    trial = handler.load_data()
    for marker in trial.markers.values():
        marker.time += .25
    for event in trial.events:
        event.time += .25
    with pytest.raises(ValueError, match='first_frame / sampling_rate'):
        handler.write_c3d(str(tmp_path / 'shifted.c3d'))


@pytest.mark.parametrize('explicit_clock', [False, True])
def test_legacy_h5_events_and_markers_use_the_recorded_origin(converted_h5, explicit_clock):
    start = 25. if explicit_clock else 5.
    with h5py.File(converted_h5, 'r+') as file:
        file['Trajectories'].attrs['StartFrame'] = 500
        group = file['Trajectories/Labeled']
        del group['Time']
        if explicit_clock:
            group.create_dataset('Time', data=start + np.arange(20) / 100)
        events = file['Events']
        events.create_dataset('Time', data=[start + .08])
        events.attrs['LABELS'] = ['contact']
    handler = H5Handler(str(converted_h5))
    trial = handler.load_data()
    marker = trial.markers['Marker']
    assert marker.first_frame == 500
    assert marker.time[0] == pytest.approx(start)
    assert trial.events[0].frame == 508
    assert trial.events[0].time == pytest.approx(marker.time[8])
    handler.save_data(trial, str(converted_h5))
    reloaded = handler.load_data()
    assert_events_equal(reloaded.events, trial.events)
    np.testing.assert_array_equal(reloaded.markers['Marker'].time, marker.time)


def test_reported_strn_index_and_time():
    from ibo_biomech.containers._validation import frame_at_time
    assert frame_at_time(1.56666666666, 120, 88, 88 / 120) == 188


def test_h5_conversion_edits_selective_load_and_subject_cache(event_c3d, tmp_path):
    path = tmp_path / 'events.h5'
    FileConverter.c3d_to_h5(str(event_c3d), str(path))
    handler = H5Handler(str(path))
    trial = handler.load_data()
    assert_events_equal(trial.events, C3DHandler(str(event_c3d)).load_data().events)
    subject = handler.load_subject_data()
    assert_events_equal(subject.get_trial_by_idx(0).events, trial.events)
    subject.id = 'P1'
    subject.save_cache(str(tmp_path))
    cached = Subject.load_from_cache(str(tmp_path / 'P1_cache.pkl'))
    assert_events_equal(cached.get_trial_by_idx(0).events, trial.events)
    trial.events.pop(0)
    trial.events.append(Event('左 contact', 6111, 61.11, description='左'))
    handler.save_data(trial, str(path))
    assert_events_equal(handler.load_data().events, trial.events)
    skipped = handler.load_data(load_events=False)
    assert skipped.events == []
    skipped.crop('markers', 5, 15)
    handler.save_data(skipped, str(path))
    assert_events_equal(handler.load_data().events, trial.events)
    with h5py.File(path) as file:
        assert 'SourceData/Events' not in file
    actual = handler.load_data()
    actual.events.clear()
    handler.save_data(actual, str(path))
    assert handler.load_data().events == []


def test_h5_event_crop_is_explicit_and_keeps_absolute_clock(event_c3d, tmp_path):
    path = tmp_path / 'crop.h5'
    FileConverter.c3d_to_h5(str(event_c3d), str(path))
    handler = H5Handler(str(path))
    trial = handler.load_data()
    trial.crop('markers', 5, 15)
    expected = list(trial.events)
    handler.save_data(trial, str(path))
    assert_events_equal(handler.load_data().events, expected)
    trial.crop_events(6105, 6115)
    handler.save_data(trial, str(path))
    assert [event.frame for event in handler.load_data().events] == [6108]


def test_legacy_h5_event_loading(converted_h5):
    with h5py.File(converted_h5, 'r+') as file:
        group = file['Events']
        group.create_dataset('Time', data=[.01, .05])
        group.attrs['LABELS'] = ['strike', 'strike']
        group.attrs['CONTEXTS'] = ['Left', 'Right']
        group.attrs['DESCRIPTIONS'] = ['first contact', 'second contact']
    handler = H5Handler(str(converted_h5))
    trial = handler.load_data()
    assert [event.frame for event in trial.events] == [1, 5]
    assert [event.description for event in trial.events] == ['first contact', 'second contact']
    handler.save_data(trial, str(converted_h5))
    assert_events_equal(handler.load_data().events, trial.events)


@pytest.mark.parametrize('corruption', ['count', 'frame', 'time', 'version', 'missing'])
def test_malformed_h5_events_rejected(event_c3d, tmp_path, corruption):
    path = tmp_path / 'bad.h5'
    FileConverter.c3d_to_h5(str(event_c3d), str(path))
    with h5py.File(path, 'r+') as file:
        group = file['Events']
        if corruption == 'count':
            del group['Time']
            group.create_dataset('Time', data=[1.])
        elif corruption == 'frame':
            del group['Frame']
            group.create_dataset('Frame', data=[1.5, 2., 3.])
        elif corruption == 'time':
            group['Time'][0] = np.nan
        elif corruption == 'version':
            group.attrs['SchemaVersion'] = 99
        else:
            del group['Name']
    with pytest.raises(ValueError):
        H5Handler(str(path)).load_data()


def test_h5_rejected_save_does_not_change_file(event_c3d, tmp_path):
    path = tmp_path / 'events.h5'
    FileConverter.c3d_to_h5(str(event_c3d), str(path))
    handler = H5Handler(str(path))
    trial = handler.load_data()
    before = path.read_bytes()
    trial.events[0].time = np.nan
    with pytest.raises(ValueError):
        handler.save_data(trial, str(path))
    assert path.read_bytes() == before


def test_event_fields_and_h5_drop_old_annotations(event_c3d, tmp_path):
    from dataclasses import fields

    assert {field.name for field in fields(Event)} == {'name', 'description', 'frame', 'time'}
    path = tmp_path / 'minimal_events.h5'
    FileConverter.c3d_to_h5(str(event_c3d), str(path))
    with h5py.File(path, 'r+') as file:
        group = file['Events']
        assert set(group) == {'Name', 'Description', 'Frame', 'Time'}
        for key in ('Context', 'Subject'):
            group.create_dataset(key, data=['old'] * 3, dtype=h5py.string_dtype('utf-8'))
        for key in ('IconID', 'GenericFlag'):
            group.create_dataset(key, data=[1, 1, 1])
    handler = H5Handler(str(path))
    trial = handler.load_data()
    assert trial.events[1].description == 'contact'
    handler.save_data(trial, str(path))
    with h5py.File(path) as file:
        assert set(file['Events']) == {'Name', 'Description', 'Frame', 'Time'}
    assert_events_equal(handler.load_data().events, trial.events)


def test_user_example_events_round_trip(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'example_data/test_c3d_events.c3d'
    if not path.exists():
        pytest.skip('Optional local recording is not checked into the repository.')
    handler = C3DHandler(str(path))
    trial = handler.load_data()
    assert [event.name for event in trial.events] == ['Foot Strike', 'Foot Strike', 'random event']
    assert [event.frame for event in trial.events] == [121, 193, 223]
    output = tmp_path / 'example.h5'
    FileConverter.c3d_to_h5(str(path), str(output))
    assert_events_equal(H5Handler(str(output)).load_data().events, trial.events)
    c3d_output = tmp_path / 'example.c3d'
    handler.write_c3d(str(c3d_output))
    assert_events_equal(C3DHandler(str(c3d_output)).load_data().events, trial.events)
