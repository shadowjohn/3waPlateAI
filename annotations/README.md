# Pose 修正點（不含圖片）

`pose-clean119-v1.json` 是羽山人工修正的 119 張車牌四角座標，依使用者同意提供。僅保留來源識別、圖片 SHA-256、寬高、validation index、修正座標與既有固定分組：83 train／36 holdout，seed 20260923。

座標為原圖 pixel centre，順序為左上、右上、右下、左下；`corners` 為 instance 陣列。請先用 hash 找到完全相同的圖片，再套座標，不要把 validation index 當上游 source index。沒有圖、縮圖、Base64、原始標註、審核筆記或本機路徑。

`pose-pilot-inputs-v1.json` 只保存原始 2,000 張訓練圖片的 source index、圖片 SHA-256 和原始角點指紋 `original_corners_sha256`（little-endian float32 bytes 的 SHA-256），供新機按固定上游 revision 還原選樣並防止標註被改動；不含圖片或原始標註座標。

來源：[EZCon/taiwan-license-plate-detection](https://huggingface.co/datasets/EZCon/taiwan-license-plate-detection/tree/ab64ba1e86615c8371e1b5617792a130d45028e8)。此處提供修正點不表示取得原圖或衍生模型的再散布／商用授權；這些權利仍須分別確認。

不能只用點訓練；圖片仍需自行取得。這批已經參與研究，不能宣稱全新盲測。完整操作見 [Pose 訓練說明](../docs/pose-training.md)。
