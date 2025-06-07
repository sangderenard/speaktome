import logging

from speaktome.faculty import detect_faculty, Faculty

logger = logging.getLogger(__name__)


def test_detect_faculty_enum():
    logger.info('test_detect_faculty_enum start')
    fac = detect_faculty()
    assert isinstance(fac, Faculty)
    logger.info('test_detect_faculty_enum end')
