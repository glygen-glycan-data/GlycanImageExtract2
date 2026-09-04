from PIL import Image
import os

MAN_PNG_DIR = "/home/linliu/titleTransfer/pdf/man_png_full"
MAN_TXT_DIR = "/home/linliu/titleTransfer/pdf/man_png_full"
NO_BORDER_TXT_DIR = "/home/linliu/titleTransfer/pdf/man_png_full/noBoard_txt"
os.makedirs(NO_BORDER_TXT_DIR, exist_ok=True)
BORDER = 100

Png_NAME = "Complex-N-glycans-are-required-for-binding-of-WFA-A-High-mannose-N-glycans-present-on_Q320"

# <--- 你要处理的那张图

def get_png_size(png_path):
    with Image.open(png_path) as img:
        return img.size   # (width, height)


def fix_yolo_txt(txt_in, txt_out, w_new, h_new, border=100):
    w_orig = w_new - 2 * border
    h_orig = h_new - 2 * border

    with open(txt_in) as fin, open(txt_out, "w") as fout:
        for line in fin:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, xc, yc, bw, bh = parts
            xc, yc, bw, bh = map(float, (xc, yc, bw, bh))

            # 1) 反归一化到 new image
            xc_abs = xc * w_new
            yc_abs = yc * h_new
            bw_abs = bw * w_new
            bh_abs = bh * h_new

            # 2) 撤销白边
            xc_abs -= border
            yc_abs -= border

            # 3) 重新归一化到 original image
            xc_new = xc_abs / w_orig
            yc_new = yc_abs / h_orig
            bw_new = bw_abs / w_orig
            bh_new = bh_abs / h_orig

            fout.write(f"{cls} {xc_new:.6f} {yc_new:.6f} {bw_new:.6f} {bh_new:.6f}\n")

png_path = os.path.join(MAN_PNG_DIR, Png_NAME + ".png")
txt_in   = os.path.join(MAN_TXT_DIR, Png_NAME + ".txt")
txt_out  = os.path.join(NO_BORDER_TXT_DIR, Png_NAME + ".txt")  # 输出同名 txt 到新目录

w_new, h_new = get_png_size(png_path)

fix_yolo_txt(
    txt_in=txt_in,
    txt_out=txt_out,
    w_new=w_new,
    h_new=h_new,
    border=BORDER
)

print("Done:", txt_out)