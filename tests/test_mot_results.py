import numpy as np
import pytest
from ibo_biomech.containers import IKResults, IDResults


def test_ikresults_read_parses_columns_and_degrees_flag(mot_file):
    ik = IKResults(filepath=mot_file)
    assert ik.unit == "deg"
    assert set(ik.columns) == {"pelvis_tilt", "hip_flexion_r"}
    assert len(ik.time) == 5
    assert np.allclose(ik.data["hip_flexion_r"].data, [10.0, 10.5, 11.0, 11.5, 12.0])


def test_idresults_read_same_parser_as_ikresults(mot_file):
    idr = IDResults(filepath=mot_file)
    assert idr.unit == "deg"
    assert set(idr.columns) == {"pelvis_tilt", "hip_flexion_r"}


def test_ikresults_to_rad_excludes_translations_and_time(mot_file):
    ik = IKResults(filepath=mot_file)
    ik.add_column("pelvis_tx", data=np.array([0.1, 0.2, 0.3, 0.4, 0.5]), unit="m")
    hip_deg = ik.data["hip_flexion_r"].data.copy()
    tx_before = ik.data["pelvis_tx"].data.copy()
    ik.to_rad()
    assert ik.unit == "rad"
    assert np.allclose(ik.data["hip_flexion_r"].data, np.deg2rad(hip_deg))
    # pelvis_tx is a translation, not an angle -- must be left untouched
    assert np.allclose(ik.data["pelvis_tx"].data, tx_before)


def test_ikresults_to_rad_noop_if_already_rad(mot_file, capsys):
    ik = IKResults(filepath=mot_file)
    ik.to_rad()
    hip_rad = ik.data["hip_flexion_r"].data.copy()
    ik.to_rad()  # second call: should print a message and do nothing
    assert np.allclose(ik.data["hip_flexion_r"].data, hip_rad)


def test_ikresults_to_deg_roundtrip(mot_file):
    ik = IKResults(filepath=mot_file)
    original = ik.data["hip_flexion_r"].data.copy()
    ik.to_rad()
    ik.to_deg()
    assert np.allclose(ik.data["hip_flexion_r"].data, original, atol=1e-9)


def test_idresults_has_no_to_deg_to_rad(mot_file):
    # IDResults never carries angle-unit semantics -- confirms it's safe to
    # NOT inherit to_deg/to_rad when collapsing into a shared base class.
    idr = IDResults(filepath=mot_file)
    assert not hasattr(idr, "to_deg")
    assert not hasattr(idr, "to_rad")


def test_ikresults_write_includes_indegrees_header(mot_file, tmp_path):
    ik = IKResults(filepath=mot_file)
    out = tmp_path / "out_ik.mot"
    ik.write(str(out))
    text = out.read_text()
    assert "inDegrees=" in text


def test_idresults_write_omits_indegrees_header(mot_file, tmp_path):
    idr = IDResults(filepath=mot_file)
    out = tmp_path / "out_id.mot"
    idr.write(str(out))
    text = out.read_text()
    # This is the quirk to preserve explicitly when merging into MotResults:
    # IDResults' writer never emits inDegrees=, IKResults' always does.
    assert "inDegrees=" not in text


def test_write_read_roundtrip_preserves_values(mot_file, tmp_path):
    ik = IKResults(filepath=mot_file)
    out = tmp_path / "roundtrip.mot"
    ik.write(str(out))
    ik2 = IKResults(filepath=str(out))
    assert np.allclose(ik2.data["hip_flexion_r"].data, ik.data["hip_flexion_r"].data)


def test_add_column_rejects_duplicate_name(mot_file):
    ik = IKResults(filepath=mot_file)
    with pytest.raises(ValueError):
        ik.add_column("hip_flexion_r", data=np.zeros(5))


def test_add_column_rejects_length_mismatch(mot_file):
    ik = IKResults(filepath=mot_file)
    with pytest.raises(ValueError):
        ik.add_column("new_col", data=np.zeros(3))


def test_crop_trims_time_and_all_columns(mot_file):
    ik = IKResults(filepath=mot_file)
    ik.crop(1, 4)
    assert len(ik.time) == 3
    for col in ik.data.values():
        assert len(col.data) == 3


@pytest.mark.parametrize("cls", [IKResults, IDResults])
def test_getitem_by_column_name_if_implemented(mot_file, cls):
    # As of the version this suite was written against, __getitem__ had just
    # been added to both IKResults and IDResults upstream but isn't in the
    # currently installed package -- skip cleanly if it's not there yet, but
    # run for real once it is. ASSUMPTION (please confirm): __getitem__ is
    # keyed by column name and returns the underlying Data container, e.g.
    # ik["hip_flexion_r"] -> Data(...). Adjust this test if that's wrong.
    obj = cls(filepath=mot_file)
    if not hasattr(obj, "__getitem__"):
        pytest.skip(f"{cls.__name__} does not implement __getitem__ in the installed version")
    first_col = obj.columns[0]
    assert obj[first_col] is obj.data[first_col]
