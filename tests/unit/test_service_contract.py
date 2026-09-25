import base64
import io

import numpy as np
import pytest
from PIL import Image


def image_bytes(fmt='PNG', size=(12, 8), **kwargs):
    out = io.BytesIO()
    Image.new('RGB', size, (220, 30, 10)).save(out, format=fmt, **kwargs)
    return out.getvalue()


def test_normalization_preserves_identity_and_unknown_characters():
    from plateai_service.text import normalize_text
    assert normalize_text(' ＡｂＣ－００１２ ') == 'ABC0012'
    assert normalize_text('軍 ＯI-04?') == '軍OI04?'
    assert normalize_text('KUA·0001') == 'KUA0001'


def test_largest_line_excludes_slogan_and_joins_same_row():
    from plateai_service.text import read_plate_line
    result = read_plate_line(['電動車', '5555', 'EAB-'], [
        [[0,0],[30,0],[30,5],[0,5]],
        [[50,10],[95,10],[95,40],[50,40]],
        [[0,10],[48,10],[48,40],[0,40]],
    ])
    assert result['text'] == 'EAB5555'
    assert result['raw_text'] == 'EAB- 5555'
    assert result['format_check']['allocation_status'] == 'unknown'
    assert result['format_check']['registration_status'] == 'not_checked'


def test_two_plausible_lines_are_not_arbitrarily_selected():
    from plateai_service.text import read_plate_line
    result = read_plate_line(['ABC-1234', 'XYZ-5678'], [
        [[0,0],[100,0],[100,20],[0,20]], [[0,30],[100,30],[100,50],[0,50]],
    ])
    assert result['status'] == 'ambiguous'
    assert result['text'] is None


def test_empty_and_special_text_remain_honest():
    from plateai_service.text import read_plate_line
    assert read_plate_line([], [])['status'] == 'unreadable'
    result = read_plate_line(['軍01234'], [[[0,0],[100,0],[100,20],[0,20]]])
    assert result['text'] == '軍01234'
    assert result['status'] == 'unverified_format'
    assert 'special_plate_unverified' in [w['code'] for w in result['warnings']]


def test_base64_accepts_plain_and_matching_data_uri():
    from plateai_service.images import decode_base64, decode_image
    data = image_bytes()
    encoded = base64.b64encode(data).decode()
    plain, mime = decode_base64(encoded)
    uri, uri_mime = decode_base64('data:image/png;base64,' + encoded)
    assert plain == uri == data and mime is None
    bgr = decode_image(uri, uri_mime)
    assert bgr.shape == (8,12,3)
    assert bgr[0,0].tolist() == [10,30,220]


@pytest.mark.parametrize('value', ['', '!!!!', 'data:text/plain;base64,YQ==', 'YQ==\n'])
def test_bad_base64_is_rejected(value):
    from plateai_service.errors import ServiceError
    from plateai_service.images import decode_base64
    with pytest.raises(ServiceError):
        decode_base64(value)


def test_mime_mismatch_and_animated_image_rejected():
    from plateai_service.errors import ServiceError
    from plateai_service.images import decode_image
    with pytest.raises(ServiceError, match='unsupported_image'):
        decode_image(image_bytes(), 'image/jpeg')
    with pytest.raises(ServiceError, match='unsupported_image'):
        decode_image(image_bytes('GIF'))
    out=io.BytesIO()
    Image.new('RGB',(10,10),'red').save(out,format='PNG',save_all=True,append_images=[Image.new('RGB',(10,10),'blue')])
    with pytest.raises(ServiceError, match='unsupported_image'):
        decode_image(out.getvalue())


def test_exif_coordinates_follow_transposed_image():
    from plateai_service.images import decode_image
    exif=Image.Exif();exif[274]=6
    result=decode_image(image_bytes('JPEG',size=(12,8),exif=exif))
    assert result.shape == (12,8,3)


def test_pixel_header_limit_precedes_load(monkeypatch):
    from plateai_service.errors import ServiceError
    import plateai_service.images as images
    monkeypatch.setattr(images,'MAX_PIXELS',50)
    with pytest.raises(ServiceError,match='image_too_large'):
        images.decode_image(image_bytes(size=(12,8)))


def test_invalid_image_is_not_no_plate():
    from plateai_service.errors import ServiceError
    from plateai_service.images import decode_image
    with pytest.raises(ServiceError,match='invalid_image'):
        decode_image(b'bad image')
