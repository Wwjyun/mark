# YOLOX COCO Annotator

YOLOX 用的 COCO 格式影像標注工具，以 PySide6 撰寫。

## 功能

- 開啟本地圖片資料夾（jpg / png / bmp / tif）
- 用滑鼠拖曳畫 bounding box
- 多類別支援，每個類別自動分配不同顏色
- 框上即時顯示類別名稱
- 匯入 / 輸出標準 COCO JSON
- 類別清單自動存檔（`annotator_config.json`），重啟不遺失
- 上次開啟的資料夾與圖片位置自動還原
- Undo（最多 50 步，Ctrl+Z）
- Image list 顯示每張圖的標注數量

## 環境需求

- Python 3.10+
- PySide6 6.7+

## 安裝

```bash
pip install -r requirements.txt
```

## 執行

```bash
python main.py
```

## 快捷鍵

| 按鍵 | 功能 |
|---|---|
| 左鍵拖曳 | 畫 bounding box |
| 滾輪 | 縮放 |
| Space + 左鍵 / 中鍵拖曳 | 平移畫面 |
| A | 上一張圖片 |
| D | 下一張圖片 |
| F | 適合視窗大小 |
| Del | 刪除選取的框 |
| Ctrl+Z | 復原（當前圖片） |
| Ctrl+S | 輸出 COCO JSON |

## 打包成 exe

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name YOLOXAnnotator main.py
```

輸出檔案在 `dist/YOLOXAnnotator.exe`。  
`annotator_config.json` 會自動建立在 exe 旁邊。

## 檔案結構

```
MARK/
├── main.py          # 入口
├── annotator.py     # 主視窗
├── graphics.py      # BBoxRectItem、AnnotatorView
├── coco_io.py       # COCO JSON import / export
├── config.py        # 類別與 session 持久化
└── requirements.txt
```

## COCO JSON 格式

輸出符合標準 COCO 格式，可直接用於 YOLOX 訓練：

```json
{
  "images":      [{ "id", "file_name", "width", "height" }],
  "annotations": [{ "id", "image_id", "category_id", "bbox", "area", "iscrowd", "segmentation" }],
  "categories":  [{ "id", "name" }]
}
```

`bbox` 格式為 `[x, y, width, height]`（左上角座標）。
