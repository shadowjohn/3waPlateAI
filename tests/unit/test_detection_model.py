from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from plateai_trainer.detection.contracts import CompositeInstance
from plateai_trainer.detection.loss import detection_loss
from plateai_trainer.detection.model import PlatePoseNet
from plateai_trainer.detection.targets import assign_detection_targets


torch.set_num_threads(1)


def instance(bbox=(200.0, 200.0, 280.0, 280.0), corners=None):
    x1, y1, x2, y2 = bbox
    return CompositeInstance(
        bbox_xyxy=bbox,
        corners_xy=np.asarray(
            corners if corners is not None else [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        ),
        source_plate={},
    )


def perfect_predictions(targets):
    predictions = torch.zeros((1, 8400, 13))
    predictions[..., 4] = 0.25
    boxes = torch.from_numpy(targets.bbox_xyxy)
    predictions[0, targets.positive_indices, :2] = (boxes[:, :2] + boxes[:, 2:]) / 2
    predictions[0, targets.positive_indices, 2:4] = boxes[:, 2:] - boxes[:, :2]
    predictions[0, targets.positive_indices, 5:] = torch.from_numpy(targets.corners_xy).flatten(1)
    return predictions


def test_plate_pose_net_emits_fixed_candidate_layout():
    candidates = PlatePoseNet()(torch.zeros((2, 3, 640, 640)))
    assert candidates.shape == (2, 8400, 13)
    assert candidates.dtype == torch.float32
    assert torch.isfinite(candidates).all()
    assert torch.all((0 <= candidates[..., 4]) & (candidates[..., 4] <= 1))
    assert torch.all(candidates[..., 2:4] > 0)


def test_untrained_dense_detector_does_not_start_with_thousands_of_positive_cells():
    torch.manual_seed(42)
    model = PlatePoseNet().eval()
    with torch.no_grad():
        predictions = model(torch.zeros((1, 3, 640, 640)))
    assert torch.count_nonzero(predictions[..., 4] >= 0.25) == 0


def test_decode_preserves_pyramid_row_major_order_and_semantic_corner_channels():
    model = PlatePoseNet().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for head in model.heads:
            head.bias[5:] = torch.tensor([-1., -2., 3., -4., 5., 6., -7., 8.])
        predictions = model(torch.zeros((1, 3, 640, 640)))[0]
    torch.testing.assert_close(predictions[[0, 1, 80, 6400, 8000], :2], torch.tensor(
        [[4., 4.], [12., 4.], [4., 12.], [8., 8.], [16., 16.]]
    ))
    torch.testing.assert_close(predictions[0, 5:], torch.tensor([-4., -12., 28., -28., 44., 52., -52., 68.]))
    torch.testing.assert_close(predictions[:, 4], torch.full((8400,), 0.5))


@pytest.mark.parametrize("shape,dtype", [((1, 1, 640, 640), torch.float32), ((1, 3, 320, 640), torch.float32), ((1, 3, 640, 640), torch.uint8)])
def test_model_rejects_inputs_outside_fixed_float32_rgb_contract(shape, dtype):
    with pytest.raises(ValueError, match="float32.*640"):
        PlatePoseNet()(torch.zeros(shape, dtype=dtype))


@pytest.mark.parametrize("short_side,level", [(63., 0), (64., 1), (127., 1), (127.5, 1), (128., 2)])
def test_assigner_uses_exact_scale_boundaries(short_side, level):
    targets = assign_detection_targets([instance((200., 200., 400., 200. + short_side))])
    assert np.unique(targets.positive_level_indices).tolist() == [level]


def test_assigner_selects_all_nine_neighbours_and_preserves_semantic_order():
    plate = instance(corners=[[202., 204.], [279., 200.], [280., 276.], [200., 280.]])
    targets = assign_detection_targets([plate])
    # P4 centre cell (15,15), row-major offset 6400, width 40.
    assert targets.positive_indices.tolist() == [6974, 6975, 6976, 7014, 7015, 7016, 7054, 7055, 7056]
    assert targets.matched_instance_indices.shape == (8400,)
    assert np.count_nonzero(targets.matched_instance_indices == 0) == 9
    np.testing.assert_array_equal(targets.corners_xy, np.repeat(plate.corners_xy[None], 9, axis=0))


def test_assigner_filters_neighbours_outside_bbox_and_clips_grid_edges():
    targets = assign_detection_targets([instance((0., 0., 10., 10.))])
    assert targets.positive_indices.tolist() == [0]
    targets = assign_detection_targets([instance((630., 630., 639., 639.))])
    assert targets.positive_indices.tolist() == [6399]


def test_assigner_breaks_ties_by_smaller_area_then_original_instance_index():
    large = instance((190., 190., 290., 290.))
    small = instance()
    targets = assign_detection_targets([large, small, small])
    assert targets.matched_instance_indices[7015] == 1
    targets = assign_detection_targets([small, small, large])
    assert targets.matched_instance_indices[7015] == 0


@pytest.mark.parametrize("bbox", [(10., 10., 10., 20.), (10., 10., 9., 20.), (0., 0., float('nan'), 20.), (-1., 0., 20., 20.), (0., 0., 641., 20.)])
def test_assigner_rejects_invalid_letterbox_boxes(bbox):
    with pytest.raises(ValueError, match="bbox"):
        assign_detection_targets([instance(bbox)])


def test_loss_uses_all_candidate_focal_objectness_and_zero_perfect_geometry_loss():
    targets = assign_detection_targets([instance()])
    result = detection_loss(perfect_predictions(targets), targets)
    # Binary focal BCE: gamma=2.0, positive alpha=0.25, negative alpha=0.75.
    expected = -(9 * 0.25 * 0.75**2 * math.log(0.25) + 8391 * 0.75 * 0.25**2 * math.log(0.75)) / 9
    assert float(result.objectness) == pytest.approx(expected)
    assert float(result.box_ciou) == pytest.approx(0, abs=1e-6)
    assert float(result.corner_smooth_l1) == 0
    torch.testing.assert_close(result.total, result.objectness + 5 * result.box_ciou + 2 * result.corner_smooth_l1)


@pytest.mark.parametrize("positive_p,negative_p", [(0.9, 0.1), (0.1, 0.9), (0.9, 0.9), (0.1, 0.1)])
def test_focal_objectness_easy_hard_values_and_gradients(positive_p, negative_p):
    targets = assign_detection_targets([instance()])
    predictions = perfect_predictions(targets)
    predictions[..., 4] = negative_p
    predictions[0, targets.positive_indices, 4] = positive_p
    predictions.requires_grad_()
    result = detection_loss(predictions, targets)
    # Independent scalar oracles, gamma=2.0 and alpha=0.25; do not call
    # production helpers or reuse its tensor weighting computation.
    positive_loss = -0.25 * (1 - positive_p)**2 * math.log(positive_p)
    negative_loss = -0.75 * negative_p**2 * math.log(1 - negative_p)
    assert float(result.objectness.detach()) == pytest.approx((9 * positive_loss + 8391 * negative_loss) / 9)
    result.objectness.backward()
    assert torch.isfinite(predictions.grad).all()
    positive_derivative = 0.25 * (2 * (1 - positive_p) * math.log(positive_p) - (1 - positive_p)**2 / positive_p)
    negative_derivative = 0.75 * (-2 * negative_p * math.log(1 - negative_p) + negative_p**2 / (1 - negative_p))
    assert float(predictions.grad[0, 7015, 4] * 9) == pytest.approx(positive_derivative)
    assert float(predictions.grad[0, 0, 4] * 9) == pytest.approx(negative_derivative)


@pytest.mark.parametrize("positive_p,negative_p", [(0., 1.), (1., 0.)])
def test_focal_objectness_saturated_probabilities_have_finite_loss_and_gradients(positive_p, negative_p):
    targets = assign_detection_targets([instance()])
    predictions = perfect_predictions(targets)
    predictions[..., 4] = negative_p
    predictions[0, targets.positive_indices, 4] = positive_p
    predictions.requires_grad_()
    result = detection_loss(predictions, targets)
    result.total.backward()
    assert torch.isfinite(result.total)
    assert torch.isfinite(predictions.grad).all()


def test_corner_loss_normalizes_xy_by_bbox_dimensions_and_ignores_negatives():
    targets = assign_detection_targets([instance((200., 200., 360., 280.))])
    predictions = perfect_predictions(targets)
    predictions[0, targets.positive_indices, 5:] += torch.tensor([80., 40.] * 4)
    predictions[0, 0, 5:] = 1e6
    result = detection_loss(predictions, targets)
    assert float(result.corner_smooth_l1) == pytest.approx(0.125)
    torch.testing.assert_close(result.total, result.objectness + 5 * result.box_ciou + 2 * result.corner_smooth_l1)


def test_ciou_includes_overlap_and_center_distance_penalty():
    targets = assign_detection_targets([instance((12., 14., 28., 26.))])
    predictions = perfect_predictions(targets)
    predictions[0, targets.positive_indices, 0] += 10
    result = detection_loss(predictions, targets)
    assert float(result.box_ciou) == pytest.approx(1 - 3 / 13 + 100 / 820)


def test_ciou_includes_aspect_ratio_penalty():
    targets = assign_detection_targets([instance((12., 14., 28., 26.))])
    predictions = perfect_predictions(targets)
    predictions[0, targets.positive_indices, 2:4] = torch.tensor([12., 16.])
    v = 4 / math.pi**2 * (math.atan(16 / 12) - math.atan(12 / 16)) ** 2
    assert float(detection_loss(predictions, targets).box_ciou) == pytest.approx(0.4 + v * v / (0.4 + v))


def test_empty_targets_produce_finite_differentiable_objectness_only_loss():
    targets = assign_detection_targets([])
    assert targets.positive_indices.size == 0
    assert np.all(targets.matched_instance_indices == -1)
    predictions = perfect_predictions(targets).requires_grad_()
    result = detection_loss(predictions, targets)
    result.total.backward()
    assert float(result.box_ciou.detach()) == 0
    assert float(result.corner_smooth_l1.detach()) == 0
    assert torch.isfinite(predictions.grad).all()
    assert predictions.grad[..., 4].abs().sum() > 0


def test_batched_loss_matches_each_images_targets_and_rejects_batch_mismatch():
    first = assign_detection_targets([instance()])
    second = assign_detection_targets([instance((10., 10., 20., 20.))])
    predictions = torch.cat([perfect_predictions(first), perfect_predictions(second)])
    result = detection_loss(predictions, [first, second])
    assert float(result.box_ciou) == pytest.approx(0, abs=1e-6)
    assert float(result.corner_smooth_l1) == 0
    with pytest.raises(ValueError, match="batch"):
        detection_loss(predictions, first)


def test_detector_loss_is_finite_and_updates_a_parameter():
    torch.manual_seed(7)
    model = PlatePoseNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    before = next(model.parameters()).detach().clone()
    targets = assign_detection_targets([instance()])
    result = detection_loss(model(torch.rand((1, 3, 640, 640))), targets)
    optimizer.zero_grad()
    result.total.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert next(model.parameters()).grad.abs().sum() > 0
    optimizer.step()
    assert math.isfinite(float(result.total.detach()))
    assert not torch.equal(before, next(model.parameters()).detach())
