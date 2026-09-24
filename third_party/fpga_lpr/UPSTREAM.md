# Upstream provenance: FPGA-LPR

- Architecture source: <https://github.com/evan6007/FPGA-LPR>, commit `574667ca7f5730d17b4b6fcda3ec568521bcbcd8`.
- Model source: <https://huggingface.co/evan6007/FPGA-LPR>, revision `51b9606b174eedbc091aec844c288a72aa9cd25b`.
- Source's stated license: MIT in the README. Model metadata states `license:mit`. The source does not include a named copyright line; none is fabricated here. Full MIT permission notice is in `MIT-LICENSE.txt`.
- Original `best_val_loss.pth` SHA-256: `70450f3e24570bf714e53d0c169ac52948e448c3945f6748d19909d525420fa4`.
- Original `lpr_model_weight.pth` SHA-256: `21cda2d7f095d958eb725ac2dc05e87cbd342aed5fae7c0cf304477fe5b5a42b`.
- Local changes: select the six model classes and `CHARS` from pinned `model_utils.py`, remove notebook/training imports, export CPM and LPRNet to ONNX, and implement a project-owned runtime adapter and manifest validation. Original `.pth` files and TLPD images are not redistributed.
- Converted CPM ONNX SHA-256: `a9170618969315f8e2b5d9a1e7ccf8d9e20c3468142bb66732a7ca101d55c67a`.
- Converted LPRNet ONNX SHA-256: `e10d1cd12874071ff40f8c7bd20f8fdd7e2a22cce0e320e5d4daf3de4f892be8`.
- Converter: `tools/build_fpga_lpr_onnx.py` (opset 17, dynamic batch, PyTorch `2.11.0+cu128` in this local parity run, ONNX Runtime `1.30.0`). It verifies three local TLPD crop tensors with `rtol=1e-3`, `atol=1e-4`; the replay is **not** a held-out accuracy estimate.

The author notebook uses OpenCV BGR plate crops. Project-facing RGB inputs must be converted to BGR before the original model path. The author's 37th `-` class is a CTC-style blank, not a printed separator.
