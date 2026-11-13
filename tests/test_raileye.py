import pytest
from railguard.raileye import read_frame, make_manifest, summarize_manifest


def test_read_original_format(tiny_raileye):
    root, image_root, sequences = tiny_raileye
    boxes = root / "anno/unified/bboxes" / sequences[0] / "00000.csv"
    params = root / "anno/unified/params" / sequences[0] / "00000.csv"
    result = read_frame(boxes, params)
    assert len(result) == 2
    assert result[0]["object_id"] == "123"
    assert result[0]["occlusion"] == 0
    assert result[0]["beyond_safety_line"] is True
    assert result[1]["beyond_safety_line"] is False


def test_row_order_alignment_and_boolean_spellings(tiny_raileye):
    root, _, sequences = tiny_raileye
    boxes = root / "anno/unified/bboxes" / sequences[0] / "00000.csv"
    params = root / "anno/unified/params" / sequences[0] / "00000.csv"
    # Params deliberately have no object ID. Values must attach to bbox rows
    # positionally, including case/whitespace and numeric boolean spellings.
    params.write_text("25, FALSE \n0, 1\n", encoding="utf-8")
    result = read_frame(boxes, params)
    assert [(row["object_id"], row["occlusion"], row["beyond_safety_line"])
            for row in result] == [("123", 25.0, False), ("124", 0.0, True)]


def test_read_empty_frame(tiny_raileye):
    root, _, sequences = tiny_raileye
    boxes = root / "anno/unified/bboxes" / sequences[0] / "00015.csv"
    params = root / "anno/unified/params" / sequences[0] / "00015.csv"
    assert read_frame(boxes, params) == []


def test_reject_bad_parameter_alignment(tiny_raileye):
    root, _, sequences = tiny_raileye
    p = root / "anno/unified/params" / sequences[0] / "00000.csv"
    p.write_text("0,True\n", encoding="utf-8")
    with pytest.raises(ValueError, match="count mismatch"):
        read_frame(root / "anno/unified/bboxes" / sequences[0] / "00000.csv", p)


def test_manifest_includes_empty_frames(tiny_raileye):
    root, images, sequences = tiny_raileye
    df = make_manifest(root, images, sequences)
    assert len(df) == 9
    assert df[df.frame == 15].object_present.eq(False).all()
    stats = summarize_manifest(df)
    assert stats["frames"] == 6
    assert stats["person_instances"] == 6
    assert stats["unsafe_person_instances"] == 3


def test_missing_image_not_silently_skipped(tiny_raileye):
    root, images, sequences = tiny_raileye
    (images / sequences[0] / "00000.png").unlink()
    with pytest.raises(FileNotFoundError, match="Image not found"):
        make_manifest(root, images)


def test_missing_image_allowed_for_audit(tiny_raileye):
    root, _, _ = tiny_raileye
    df = make_manifest(root, image_root=None, require_images=False)
    assert df.image.eq("").all()


def test_missing_parameter_file_not_silently_skipped(tiny_raileye):
    root, images, sequences = tiny_raileye
    (root / "anno/unified/params" / sequences[0] / "00000.csv").unlink()
    with pytest.raises(FileNotFoundError, match="Missing paired parameter file"):
        make_manifest(root, images)


def test_invalid_boolean_is_rejected_with_row_context(tiny_raileye):
    root, _, sequences = tiny_raileye
    boxes = root / "anno/unified/bboxes" / sequences[0] / "00000.csv"
    params = root / "anno/unified/params" / sequences[0] / "00000.csv"
    params.write_text("0,maybe\n25,False\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"row 1: invalid boolean"):
        read_frame(boxes, params)
