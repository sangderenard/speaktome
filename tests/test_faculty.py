from speaktome.faculty import detect_faculty, Faculty


def test_detect_faculty_enum():
    fac = detect_faculty()
    assert isinstance(fac, Faculty)
