# file: cc_sim_allfolders.py
# ------------------------------------------------------
# 遍历 results 根目录下所有子文件夹（如 000, 001, 002...）
# 对每个子文件夹的预测图与真实显著性图逐帧计算 CC / SIM
# 每个子文件夹输出独立 TXT 文件
# ------------------------------------------------------

from pathlib import Path
import numpy as np
from PIL import Image

# ========== 1. 路径与参数（按需修改） ==========
# 修改为你自己的预测图与真实显著性图路径
RESULTS_ROOT = Path(r"E:\Others\SST-Sal\SST4\SST-Sal-main\data\results\yuce")  # 预测显著性图所在文件夹
GT_ROOT = Path(r"E:\Others\SST-Sal\SST4\SST-Sal-main\data\results\zhenshi")  # 真实显著性图文件夹
OUT_ROOT = Path(r"E:\Others\SST-Sal\SST4\SST-Sal-main\data\cc_sim_000.txt")  # 输出TXT路径

# 支持的图片扩展名
EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

# ========== 2. 工具函数 ==========
def read_gray_01(path: Path) -> np.ndarray:
    """读取单通道灰度图并转为 [0,1] float32。"""
    im = Image.open(path).convert("L")
    arr = np.asarray(im, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr

def resize_to(img: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """双线性插值到目标尺寸 (H, W)。"""
    H, W = target_hw
    pil = Image.fromarray((img * 255).astype(np.uint8))
    pil = pil.resize((W, H), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0

def cc(pred: np.ndarray, gt: np.ndarray) -> float:
    """线性相关系数 CC，范围 [-1,1]，越大越好。"""
    P = pred.astype(np.float64).ravel()
    G = gt.astype(np.float64).ravel()
    mu_p, mu_g = P.mean(), G.mean()
    std_p, std_g = P.std(), G.std()
    if std_p < 1e-8 or std_g < 1e-8:
        return 0.0
    cov = np.mean((P - mu_p) * (G - mu_g))
    return float(cov / (std_p * std_g))

def sim(pred: np.ndarray, gt: np.ndarray) -> float:
    """相似性度量 SIM，范围 [0,1]，越大越好。"""
    P = np.clip(pred.astype(np.float64), 0.0, None)
    G = np.clip(gt.astype(np.float64), 0.0, None)
    sP, sG = P.sum(), G.sum()
    if sP <= 0: P = np.ones_like(P); sP = P.sum()
    if sG <= 0: G = np.ones_like(G); sG = G.sum()
    P /= sP
    G /= sG
    return float(np.minimum(P, G).sum())

def list_by_ext(d: Path, exts: list[str]) -> set[str]:
    """列出目录下指定后缀的文件名集合（包含扩展名）。"""
    names = set()
    for ext in exts:
        for p in d.glob(f"*{ext}"):
            names.add(p.name)
    return names

# ========== 3. 单个子文件夹处理函数 ==========
def process_folder(pred_dir: Path, gt_dir: Path, out_txt: Path):
    """对单个子文件夹进行逐帧对比并保存结果。"""
    if not gt_dir.exists():
        print(f"⚠️ 真实文件夹缺失: {gt_dir}")
        return None

    pred_names = list_by_ext(pred_dir, EXTS)
    gt_names   = list_by_ext(gt_dir, EXTS)
    inter = sorted(pred_names & gt_names)
    if not inter:
        print(f"⚠️ 无匹配图像: {pred_dir.name}")
        return None

    cc_list, sim_list = [], []
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("# filename\tCC\tSIM\n")
        for name in inter:
            pred_path = pred_dir / name
            gt_path   = gt_dir / name
            pred = read_gray_01(pred_path)
            gt   = read_gray_01(gt_path)
            if pred.shape != gt.shape:
                pred = resize_to(pred, gt.shape)
            cc_val  = cc(pred, gt)
            sim_val = sim(pred, gt)
            cc_list.append(cc_val)
            sim_list.append(sim_val)
            f.write(f"{name}\t{cc_val:.6f}\t{sim_val:.6f}\n")

        cc_mean, sim_mean = float(np.mean(cc_list)), float(np.mean(sim_list))
        f.write(f"\n# AVG_CC\t{cc_mean:.6f}\n# AVG_SIM\t{sim_mean:.6f}\n")

    print(f"✅ {pred_dir.name}: 平均CC={cc_mean:.4f}, 平均SIM={sim_mean:.4f}, 图像数={len(inter)}")
    return cc_mean, sim_mean

# ========== 4. 主循环：遍历所有子文件夹 ==========
def main():
    if not RESULTS_ROOT.is_dir():
        print("❌ 预测结果目录不存在，请检查路径。")
        return

    subfolders = [d for d in RESULTS_ROOT.iterdir() if d.is_dir()]
    if not subfolders:
        print("⚠️ 未发现子文件夹。")
        return

    all_cc, all_sim = [], []
    for sub in sorted(subfolders):
        gt_sub = GT_ROOT / sub.name
        out_txt = OUT_ROOT / f"{sub.name}_cc_sim.txt"
        res = process_folder(sub, gt_sub, out_txt)
        if res is not None:
            all_cc.append(res[0])
            all_sim.append(res[1])

    if all_cc:
        print("\n==============================")
        print(f"🌐 全部平均 CC = {np.mean(all_cc):.6f}")
        print(f"🌐 全部平均 SIM = {np.mean(all_sim):.6f}")
        print("==============================")

if __name__ == "__main__":
    main()