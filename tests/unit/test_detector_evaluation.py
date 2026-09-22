def test_bbox_operating_metrics_are_separate_from_corner_and_ocr_quality():
    from tools.evaluate_detector import operating_summary
    result = operating_summary({'instances': 10, 'corner_matched_instances': 9,
        'nms_predictions': 12, 'complete_quad_recall': .7})
    assert result['bbox_precision'] == .75
    assert result['bbox_recall'] == .9
    assert result['local_promotion_gate'] is False
    assert operating_summary({'instances': 10, 'corner_matched_instances': 10,
        'nms_predictions': 10, 'complete_quad_recall': .8})['local_promotion_gate'] is True


def test_empty_predictions_do_not_satisfy_promotion_gate():
    from tools.evaluate_detector import operating_summary
    result = operating_summary({'instances': 0, 'corner_matched_instances': 0,
        'nms_predictions': 0, 'complete_quad_recall': 0})
    assert result['bbox_precision'] == result['bbox_recall'] == 0
    assert result['local_promotion_gate'] is False
