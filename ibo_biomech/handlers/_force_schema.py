"""Current HDF5 force schema, shared by conversion, loading and saving.

Only this schema is supported. Tz is a full moment-at-CoP vector with shape
(3, n_samples), in the declared force/CoP coordinate frame. Wholly NaN geometry
means unknown. Older files must be regenerated from their source recordings.
"""
from ibo_biomech.containers import ForceData


VERSION = 2
FIELDS = {'Force': 'force', 'Moment': 'moment', 'COP': 'cop', 'Tz': 'Tz',
          'Corners': 'corners', 'Origin': 'origin', 'Position': 'position',
          'Rotation': 'rotation', 'Time': 'time'}


def read_plate(group):
    """Require the current schema and validate every stored array shape."""
    if group.attrs.get('SchemaVersion') != VERSION:
        raise ValueError(f'{group.name}: only force schema {VERSION} is supported; '
                         'regenerate the HDF5 file from its source recording.')
    required = set(FIELDS) - {'Time'}
    missing = required - set(group)
    if missing:
        raise ValueError(f'{group.name}: missing required datasets: {sorted(missing)}.')
    values = {field: group[key][:] if key in group else None for key, field in FIELDS.items()}
    force = values['force']
    if force.ndim != 2 or force.shape[0] != 3:
        raise ValueError(f'{group.name}: Force must have shape (3, n_samples).')
    n = force.shape[1]
    if group.attrs.get('NumSamples') != n:
        raise ValueError(f'{group.name}: NumSamples does not match Force.')
    for field, shape in [('moment', (3, n)), ('cop', (3, n)), ('Tz', (3, n)),
                         ('corners', (3, 4, n)), ('origin', (3, 1)),
                         ('position', (3, n)), ('rotation', (3, 3, n))]:
        if values[field].shape != shape:
            raise ValueError(f'{group.name}: {field} must have shape {shape}; '
                             'regenerate incompatible files from their source recordings.')
    metadata = {key: group.attrs.get(key, 'Unknown')
                for key in ('unit_force', 'unit_moment', 'unit_position')}
    if 'CoordinateSystem' not in group.attrs:
        raise ValueError(f'{group.name}: CoordinateSystem must declare the vector frame.')
    expected_frame = 'global' if group.attrs['CoordinateSystem'] else 'local'
    if group.attrs.get('FreeMomentFrame') != expected_frame:
        raise ValueError(f'{group.name}: FreeMomentFrame must match CoordinateSystem.')
    name = group.attrs.get('Name', f'ForcePlate_{group.name.rsplit("/", 1)[-1]}')
    if isinstance(name, bytes):
        name = name.decode()
    return ForceData(name=name, metadata=metadata, **values,
                     sampling_rate=group.attrs.get('SamplingFrequency'),
                     coordinateSystem=group.attrs['CoordinateSystem'])


def write_plate(group, plate):
    """Replace current-schema fields and refresh units/frame/sample metadata."""
    rate = plate.validate()
    for key, field in FIELDS.items():
        if key in group:
            del group[key]
        value = getattr(plate, field)
        if value is not None:
            group.create_dataset(key, data=value, compression='gzip')
    group.attrs.update(SchemaVersion=VERSION, Name=plate.name,
                       CoordinateSystem=int(plate.coordinateSystem),
                       FreeMomentFrame='global' if plate.coordinateSystem else 'local',
                       NumSamples=plate.num_samples, unit_force=plate.unit_force,
                       unit_moment=plate.unit_moment, unit_position=plate.unit_cop)
    for key in ('SamplingFrequency', 'SamplingFactor'):
        if key in group.attrs:
            del group.attrs[key]
    if rate is not None:
        group.attrs['SamplingFrequency'] = rate
