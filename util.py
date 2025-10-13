from PIL import Image
from io import BytesIO
import base64
import cv2
# -------------------- 유틸 --------------------
def crop_roi(bgr, x1, y1, x2, y2, margin=0.25):
    h, w = bgr.shape[:2]
    dx = int((x2 - x1) * margin)
    dy = int((y2 - y1) * margin)
    X1 = max(0, x1 - dx)
    Y1 = max(0, y1 - dy)
    X2 = min(w - 1, x2 + dx)
    Y2 = min(h - 1, y2 + dy)
    if X2 <= X1 or Y2 <= Y1:
        return None
    return bgr[Y1:Y2, X1:X2].copy(), [X1, Y1, X2, Y2]

def bgr_to_b64jpg(bgr, quality=80):
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")