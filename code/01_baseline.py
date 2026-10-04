from pathlib import Path
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# =========================================================
# 1. 路径与实验设置
# =========================================================

project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data" / "Set12"
result_dir = project_dir / "results"

result_dir.mkdir(exist_ok=True)

# 先选三张结构差异明显的图片
image_names = [
    "01.png",
    "02.png",
    "03.png",
    "04.png",
    "05.png",
    "06.png",
    "07.png",
    "08.png",
    "09.png",
    "10.png",
    "11.png",
    "12.png"
]
# 高斯噪声强度
sigma_n = 25

# 固定随机种子，保证实验可重复
rng = np.random.default_rng(seed=42)


# =========================================================
# 2. 评价函数
# =========================================================

def evaluate(reference, test):

    psnr = peak_signal_noise_ratio(
        reference,
        test,
        data_range=255
    )

    ssim = structural_similarity(
        reference,
        test,
        data_range=255
    )

    return psnr, ssim


# =========================================================
# 3. 保存所有实验指标
# =========================================================

all_results = []


# =========================================================
# 4. 依次处理三张图片
# =========================================================

for image_name in image_names:

    print("\n" + "=" * 50)
    print(f"正在处理：{image_name}")
    print("=" * 50)

    image_path = data_dir / image_name

    # ---------- 读取原图 ----------
    original = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if original is None:
        raise FileNotFoundError(
            f"找不到图片：{image_path}"
        )

    print("图片尺寸：", original.shape)


    # ---------- 添加高斯噪声 ----------
    noise = rng.normal(
        loc=0,
        scale=sigma_n,
        size=original.shape
    )

    noisy = original.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255)
    noisy = noisy.astype(np.uint8)


    # ---------- 高斯滤波 ----------
    gaussian = cv2.GaussianBlur(
        noisy,
        (5, 5),
        1.2
    )


    # ---------- 中值滤波 ----------
    median = cv2.medianBlur(
        noisy,
        5
    )


    # ---------- 双边滤波 ----------
    bilateral = cv2.bilateralFilter(
        noisy,
        d=9,
        sigmaColor=50,
        sigmaSpace=5
    )


    # ---------- 汇总方法 ----------
    methods = {
        "Noisy": noisy,
        "Gaussian": gaussian,
        "Median": median,
        "Bilateral": bilateral
    }


    # ---------- 评价 ----------
    print("\n评价结果：")

    for method_name, image in methods.items():

        psnr, ssim = evaluate(
            original,
            image
        )

        print(
            f"{method_name:10s} "
            f"PSNR = {psnr:6.2f} dB   "
            f"SSIM = {ssim:.4f}"
        )

        all_results.append({
            "Image": image_name,
            "Method": method_name,
            "PSNR": psnr,
            "SSIM": ssim
        })


    # ---------- 保存单独结果图 ----------
    stem = Path(image_name).stem

    cv2.imwrite(
        str(result_dir / f"{stem}_noisy.png"),
        noisy
    )

    cv2.imwrite(
        str(result_dir / f"{stem}_gaussian.png"),
        gaussian
    )

    cv2.imwrite(
        str(result_dir / f"{stem}_median.png"),
        median
    )

    cv2.imwrite(
        str(result_dir / f"{stem}_bilateral.png"),
        bilateral
    )


    # ---------- 保存五图比较 ----------
    images = [
        original,
        noisy,
        gaussian,
        median,
        bilateral
    ]

    titles = [
        "Original",
        "Noisy",
        "Gaussian",
        "Median",
        "Bilateral"
    ]

    plt.figure(figsize=(15, 4))

    for i, (image, title) in enumerate(
        zip(images, titles),
        start=1
    ):
        plt.subplot(1, 5, i)

        plt.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        plt.title(title)
        plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        result_dir / f"{stem}_comparison.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# =========================================================
# 5. 保存指标表 CSV
# =========================================================

csv_path = result_dir / "baseline_metrics.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Image",
            "Method",
            "PSNR",
            "SSIM"
        ]
    )

    writer.writeheader()
    writer.writerows(all_results)


print("\n" + "=" * 50)
print("三张图片全部处理完成")
print(f"指标表已保存：{csv_path}")
print("=" * 50)
# =========================================================
# 6. Bilateral 参数实验：只改变 sigmaColor
# =========================================================

print("\n")
print("=" * 60)
print("Bilateral sigmaColor 参数实验")
print("=" * 60)

sigma_color_values = [40, 50, 60, 70, 80]

parameter_results = []

# 先继续用 01、04、08 三张图
for image_name in image_names:

    image_path = data_dir / image_name

    original = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    # 为了参数比较公平：
    # 每张图重新固定随机种子
for idx, image_name in enumerate(image_names):

    rng_param = np.random.default_rng(
        seed=42 + idx
    )

    noise = rng_param.normal(
        loc=0,
        scale=sigma_n,
        size=original.shape
    )

    noisy = original.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    print(f"\n图片：{image_name}")

    for sigma_color in sigma_color_values:

        bilateral_test = cv2.bilateralFilter(
            noisy,
            d=9,
            sigmaColor=sigma_color,
            sigmaSpace=5
        )

        psnr, ssim = evaluate(
            original,
            bilateral_test
        )

        print(
            f"sigmaColor={sigma_color:3d}   "
            f"PSNR={psnr:6.2f} dB   "
            f"SSIM={ssim:.4f}"
        )

        parameter_results.append({
            "Image": image_name,
            "sigmaColor": sigma_color,
            "PSNR": psnr,
            "SSIM": ssim
        })


# 保存参数实验数据
parameter_csv = result_dir / "bilateral_sigmaColor_test.csv"

with open(
    parameter_csv,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Image",
            "sigmaColor",
            "PSNR",
            "SSIM"
        ]
    )

    writer.writeheader()
    writer.writerows(parameter_results)

print("\n参数实验完成")
print(f"结果保存到：{parameter_csv}")
# =========================================================
# 7. 统计不同 sigmaColor 的平均性能
# =========================================================

print("\n")
print("=" * 60)
print("不同 sigmaColor 的平均性能")
print("=" * 60)

for sigma_color in sigma_color_values:

    current = [
        r for r in parameter_results
        if r["sigmaColor"] == sigma_color
    ]

    mean_psnr = np.mean([
        r["PSNR"] for r in current
    ])

    mean_ssim = np.mean([
        r["SSIM"] for r in current
    ])

    print(
        f"sigmaColor={sigma_color:3d}   "
        f"平均PSNR={mean_psnr:6.2f} dB   "
        f"平均SSIM={mean_ssim:.4f}"
    )
    # =========================================================
# 8. 绘制平均参数曲线
# =========================================================

mean_psnr_list = []
mean_ssim_list = []

for sigma_color in sigma_color_values:

    current = [
        r for r in parameter_results
        if r["sigmaColor"] == sigma_color
    ]

    mean_psnr_list.append(
        np.mean([r["PSNR"] for r in current])
    )

    mean_ssim_list.append(
        np.mean([r["SSIM"] for r in current])
    )


# PSNR 曲线
plt.figure(figsize=(6, 4))

plt.plot(
    sigma_color_values,
    mean_psnr_list,
    marker="o"
)

plt.xlabel("sigmaColor")
plt.ylabel("Average PSNR (dB)")
plt.title("Bilateral: sigmaColor vs Average PSNR")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    result_dir / "sigmaColor_average_PSNR.png",
    dpi=200
)

plt.close()


# SSIM 曲线
plt.figure(figsize=(6, 4))

plt.plot(
    sigma_color_values,
    mean_ssim_list,
    marker="o"
)

plt.xlabel("sigmaColor")
plt.ylabel("Average SSIM")
plt.title("Bilateral: sigmaColor vs Average SSIM")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    result_dir / "sigmaColor_average_SSIM.png",
    dpi=200
)

plt.close()
image_names
data_dir
result_dir
sigma_n
evaluate
# =========================================================
# 继续实验：固定 sigmaColor=60，只改变 sigmaSpace
# =========================================================

print("\n")
print("=" * 60)
print("Bilateral sigmaSpace 参数实验")
print("固定条件：d=9, sigmaColor=60")
print("=" * 60)

fixed_sigma_color = 60

sigma_space_values = [1, 2, 3, 5, 8]

space_results = []


# ---------------------------------------------------------
# 对 Set12 中当前 image_names 包含的图片逐张实验
# ---------------------------------------------------------

for idx, image_name in enumerate(image_names):

    image_path = data_dir / image_name

    original = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if original is None:
        raise FileNotFoundError(
            f"找不到图片：{image_path}"
        )

    # 每张图片使用不同、但可重复的噪声
    rng_space = np.random.default_rng(
        seed=42 + idx
    )

    noise = rng_space.normal(
        loc=0,
        scale=sigma_n,
        size=original.shape
    )

    noisy = original.astype(np.float32) + noise

    noisy = np.clip(
        noisy,
        0,
        255
    ).astype(np.uint8)

    print(f"\n图片：{image_name}")

    # -----------------------------------------------------
    # 只改变 sigmaSpace
    # -----------------------------------------------------

    for sigma_space in sigma_space_values:

        bilateral_test = cv2.bilateralFilter(
            noisy,
            d=9,
            sigmaColor=fixed_sigma_color,
            sigmaSpace=sigma_space
        )

        psnr, ssim = evaluate(
            original,
            bilateral_test
        )

        print(
            f"sigmaSpace={sigma_space:2d}   "
            f"PSNR={psnr:6.2f} dB   "
            f"SSIM={ssim:.4f}"
        )

        space_results.append({
            "Image": image_name,
            "sigmaSpace": sigma_space,
            "PSNR": psnr,
            "SSIM": ssim
        })


# =========================================================
# 计算各 sigmaSpace 的平均性能
# =========================================================

print("\n")
print("=" * 60)
print("不同 sigmaSpace 的平均性能")
print("=" * 60)

mean_space_psnr = []
mean_space_ssim = []

for sigma_space in sigma_space_values:

    current = [
        r for r in space_results
        if r["sigmaSpace"] == sigma_space
    ]

    mean_psnr = np.mean([
        r["PSNR"] for r in current
    ])

    mean_ssim = np.mean([
        r["SSIM"] for r in current
    ])

    mean_space_psnr.append(mean_psnr)
    mean_space_ssim.append(mean_ssim)

    print(
        f"sigmaSpace={sigma_space:2d}   "
        f"平均PSNR={mean_psnr:6.2f} dB   "
        f"平均SSIM={mean_ssim:.4f}"
    )


# =========================================================
# 保存 sigmaSpace 实验结果
# =========================================================

space_csv = (
    result_dir
    / "bilateral_sigmaSpace_test.csv"
)

with open(
    space_csv,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Image",
            "sigmaSpace",
            "PSNR",
            "SSIM"
        ]
    )

    writer.writeheader()
    writer.writerows(space_results)


# =========================================================
# 绘制 sigmaSpace - PSNR 曲线
# =========================================================

plt.figure(figsize=(6, 4))

plt.plot(
    sigma_space_values,
    mean_space_psnr,
    marker="o"
)

plt.xlabel("sigmaSpace")
plt.ylabel("Average PSNR (dB)")
plt.title(
    "Bilateral: sigmaSpace vs Average PSNR"
)

plt.grid(True)
plt.tight_layout()

plt.savefig(
    result_dir
    / "sigmaSpace_average_PSNR.png",
    dpi=200
)

plt.close()


# =========================================================
# 绘制 sigmaSpace - SSIM 曲线
# =========================================================

plt.figure(figsize=(6, 4))

plt.plot(
    sigma_space_values,
    mean_space_ssim,
    marker="o"
)

plt.xlabel("sigmaSpace")
plt.ylabel("Average SSIM")
plt.title(
    "Bilateral: sigmaSpace vs Average SSIM"
)

plt.grid(True)
plt.tight_layout()

plt.savefig(
    result_dir
    / "sigmaSpace_average_SSIM.png",
    dpi=200
)

plt.close()


print("\nsigmaSpace 参数实验完成")
print(f"结果保存至：{space_csv}")
# =========================================================
# Bilateral 局部二维参数检查
# sigmaColor × sigmaSpace
# =========================================================

print("\n")
print("=" * 60)
print("Bilateral 局部二维参数检查")
print("=" * 60)

sigma_color_grid = [50, 60, 70]
sigma_space_grid = [2, 3, 5]

grid_results = []

for idx, image_name in enumerate(image_names):

    image_path = data_dir / image_name

    original = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    # 每张图片固定自己的噪声
    rng_grid = np.random.default_rng(
        seed=42 + idx
    )

    noise = rng_grid.normal(
        loc=0,
        scale=sigma_n,
        size=original.shape
    )

    noisy = original.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    for sigma_color in sigma_color_grid:

        for sigma_space in sigma_space_grid:

            result = cv2.bilateralFilter(
                noisy,
                d=9,
                sigmaColor=sigma_color,
                sigmaSpace=sigma_space
            )

            psnr, ssim = evaluate(
                original,
                result
            )

            grid_results.append({
                "Image": image_name,
                "sigmaColor": sigma_color,
                "sigmaSpace": sigma_space,
                "PSNR": psnr,
                "SSIM": ssim
            })


# =========================================================
# 计算每组参数的平均结果
# =========================================================

print("\n平均结果：")

for sigma_color in sigma_color_grid:

    for sigma_space in sigma_space_grid:

        current = [
            r for r in grid_results
            if r["sigmaColor"] == sigma_color
            and r["sigmaSpace"] == sigma_space
        ]

        mean_psnr = np.mean([
            r["PSNR"] for r in current
        ])

        mean_ssim = np.mean([
            r["SSIM"] for r in current
        ])

        print(
            f"sigmaColor={sigma_color:2d}, "
            f"sigmaSpace={sigma_space:1d}   "
            f"平均PSNR={mean_psnr:6.2f} dB   "
            f"平均SSIM={mean_ssim:.4f}"
        )
# =========================================================
# 不同噪声强度下 sigmaColor 参数实验
# 固定 d=9, sigmaSpace=2
# =========================================================

print("\n")
print("=" * 65)
print("不同噪声强度下的 sigmaColor 实验")
print("固定条件：d=9, sigmaSpace=2")
print("=" * 65)

noise_sigma_values = [10, 25, 50]

sigma_color_noise_values = [
    20, 40, 60, 80, 100
]

fixed_sigma_space = 2

noise_parameter_results = []


# =========================================================
# 三种噪声强度
# =========================================================

for noise_sigma in noise_sigma_values:

    print("\n")
    print("-" * 65)
    print(f"高斯噪声 sigma_n = {noise_sigma}")
    print("-" * 65)

    # -----------------------------------------------------
    # 遍历 Set12
    # -----------------------------------------------------

    for idx, image_name in enumerate(image_names):

        image_path = data_dir / image_name

        original = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        # 每张图自己的随机噪声
        # 同时保证实验可重复
        #
        # 加上 noise_sigma，
        # 让不同噪声水平不是同一份随机数简单缩放
        rng_noise = np.random.default_rng(
            seed=42 + idx + noise_sigma * 100
        )

        noise = rng_noise.normal(
            loc=0,
            scale=noise_sigma,
            size=original.shape
        )

        noisy = (
            original.astype(np.float32)
            + noise
        )

        noisy = np.clip(
            noisy,
            0,
            255
        ).astype(np.uint8)


        # -------------------------------------------------
        # 当前噪声水平下扫描 sigmaColor
        # -------------------------------------------------

        for sigma_color in sigma_color_noise_values:

            result = cv2.bilateralFilter(
                noisy,
                d=9,
                sigmaColor=sigma_color,
                sigmaSpace=fixed_sigma_space
            )

            psnr, ssim = evaluate(
                original,
                result
            )

            noise_parameter_results.append({
                "NoiseSigma": noise_sigma,
                "Image": image_name,
                "sigmaColor": sigma_color,
                "PSNR": psnr,
                "SSIM": ssim
            })


# =========================================================
# 统计每个噪声水平下不同 sigmaColor 的平均表现
# =========================================================

print("\n")
print("=" * 65)
print("不同噪声强度 × sigmaColor 的平均性能")
print("=" * 65)

for noise_sigma in noise_sigma_values:

    print(f"\n高斯噪声 sigma_n = {noise_sigma}")

    for sigma_color in sigma_color_noise_values:

        current = [
            r for r in noise_parameter_results
            if r["NoiseSigma"] == noise_sigma
            and r["sigmaColor"] == sigma_color
        ]

        mean_psnr = np.mean([
            r["PSNR"] for r in current
        ])

        mean_ssim = np.mean([
            r["SSIM"] for r in current
        ])

        print(
            f"sigmaColor={sigma_color:3d}   "
            f"平均PSNR={mean_psnr:6.2f} dB   "
            f"平均SSIM={mean_ssim:.4f}"
        )


# =========================================================
# 保存 CSV
# =========================================================

noise_csv = (
    result_dir
    / "noise_level_sigmaColor_test.csv"
)

with open(
    noise_csv,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "NoiseSigma",
            "Image",
            "sigmaColor",
            "PSNR",
            "SSIM"
        ]
    )

    writer.writeheader()
    writer.writerows(
        noise_parameter_results
    )

print("\n噪声强度实验完成")
print(f"结果保存至：{noise_csv}")
# =========================================================
# 强噪声 sigma_n=50 下继续寻找 sigmaColor 峰值
# 固定 d=9, sigmaSpace=2
# =========================================================

print("\n")
print("=" * 65)
print("sigma_n=50：继续扩大 sigmaColor 范围")
print("=" * 65)

strong_noise_sigma = 50

strong_sigma_colors = [
    100, 120, 140, 160, 180, 200
]

strong_results = []


for idx, image_name in enumerate(image_names):

    image_path = data_dir / image_name

    original = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    rng_strong = np.random.default_rng(
        seed=42 + idx + strong_noise_sigma * 100
    )

    noise = rng_strong.normal(
        loc=0,
        scale=strong_noise_sigma,
        size=original.shape
    )

    noisy = original.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)


    for sigma_color in strong_sigma_colors:

        result = cv2.bilateralFilter(
            noisy,
            d=9,
            sigmaColor=sigma_color,
            sigmaSpace=2
        )

        psnr, ssim = evaluate(
            original,
            result
        )

        strong_results.append({
            "Image": image_name,
            "sigmaColor": sigma_color,
            "PSNR": psnr,
            "SSIM": ssim
        })


print("\nsigma_n=50 的扩大范围结果：")

for sigma_color in strong_sigma_colors:

    current = [
        r for r in strong_results
        if r["sigmaColor"] == sigma_color
    ]

    mean_psnr = np.mean([
        r["PSNR"] for r in current
    ])

    mean_ssim = np.mean([
        r["SSIM"] for r in current
    ])

    print(
        f"sigmaColor={sigma_color:3d}   "
        f"平均PSNR={mean_psnr:6.2f} dB   "
        f"平均SSIM={mean_ssim:.4f}"
    )
# =========================================================
# 固定参数 vs 噪声自适应参数
# =========================================================

print("\n")
print("=" * 70)
print("固定 sigmaColor vs 噪声自适应 sigmaColor")
print("=" * 70)

noise_levels = [10, 25, 50]

# 固定方案
fixed_sigma_color = 60

# 根据前面实验得到的 PSNR 较优参数
adaptive_sigma_color = {
    10: 20,
    25: 60,
    50: 160
}

fixed_results = []
adaptive_results = []


for noise_sigma in noise_levels:

    for idx, image_name in enumerate(image_names):

        image_path = data_dir / image_name

        original = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        # 固定随机种子，保证两种方案面对同一份噪声
        rng_compare = np.random.default_rng(
            seed=42 + idx + noise_sigma * 100
        )

        noise = rng_compare.normal(
            loc=0,
            scale=noise_sigma,
            size=original.shape
        )

        noisy = original.astype(np.float32) + noise
        noisy = np.clip(
            noisy,
            0,
            255
        ).astype(np.uint8)


        # =================================================
        # 方案1：固定 sigmaColor = 60
        # =================================================

        fixed_img = cv2.bilateralFilter(
            noisy,
            d=9,
            sigmaColor=fixed_sigma_color,
            sigmaSpace=2
        )

        fixed_psnr, fixed_ssim = evaluate(
            original,
            fixed_img
        )

        fixed_results.append({
            "NoiseSigma": noise_sigma,
            "PSNR": fixed_psnr,
            "SSIM": fixed_ssim
        })


        # =================================================
        # 方案2：根据噪声强度选择 sigmaColor
        # =================================================

        current_sigma_color = adaptive_sigma_color[
            noise_sigma
        ]

        adaptive_img = cv2.bilateralFilter(
            noisy,
            d=9,
            sigmaColor=current_sigma_color,
            sigmaSpace=2
        )

        adaptive_psnr, adaptive_ssim = evaluate(
            original,
            adaptive_img
        )

        adaptive_results.append({
            "NoiseSigma": noise_sigma,
            "PSNR": adaptive_psnr,
            "SSIM": adaptive_ssim
        })


# =========================================================
# 输出比较结果
# =========================================================

print("\n平均比较：")

for noise_sigma in noise_levels:

    fixed_current = [
        r for r in fixed_results
        if r["NoiseSigma"] == noise_sigma
    ]

    adaptive_current = [
        r for r in adaptive_results
        if r["NoiseSigma"] == noise_sigma
    ]

    fixed_psnr = np.mean([
        r["PSNR"] for r in fixed_current
    ])

    fixed_ssim = np.mean([
        r["SSIM"] for r in fixed_current
    ])

    adaptive_psnr = np.mean([
        r["PSNR"] for r in adaptive_current
    ])

    adaptive_ssim = np.mean([
        r["SSIM"] for r in adaptive_current
    ])

    print(f"\n噪声 sigma_n={noise_sigma}")

    print(
        f"固定参数 r=60：   "
        f"PSNR={fixed_psnr:.2f} dB   "
        f"SSIM={fixed_ssim:.4f}"
    )

    print(
        f"自适应参数 r={adaptive_sigma_color[noise_sigma]}： "
        f"PSNR={adaptive_psnr:.2f} dB   "
        f"SSIM={adaptive_ssim:.4f}"
    )

    print(
        f"PSNR提升："
        f"{adaptive_psnr-fixed_psnr:+.2f} dB   "
        f"SSIM变化："
        f"{adaptive_ssim-fixed_ssim:+.4f}"
    )
# =========================================================
# Bilateral 二维参数联合搜索
# sigmaColor × sigmaSpace
# =========================================================

print("\n")
print("=" * 80)
print("Bilateral 二维参数联合搜索")
print("=" * 80)


# 不同噪声水平对应搜索范围
search_config = {

    10: {
        "sigmaColor": [10, 20, 30, 40],
        "sigmaSpace": [1, 2, 3, 5]
    },

    25: {
        "sigmaColor": [40, 50, 60, 70, 80],
        "sigmaSpace": [1, 2, 3, 5]
    },

    50: {
        "sigmaColor": [120, 140, 160, 180, 200],
        "sigmaSpace": [1, 2, 3, 5]
    }
}


grid_search_results = []


# =========================================================
# 不同噪声水平
# =========================================================

for noise_sigma in search_config.keys():

    print("\n")
    print("-" * 80)
    print(f"当前噪声水平 sigma_n={noise_sigma}")
    print("-" * 80)


    sigma_color_values = (
        search_config[noise_sigma]["sigmaColor"]
    )

    sigma_space_values = (
        search_config[noise_sigma]["sigmaSpace"]
    )


    # -----------------------------------------------------
    # 遍历图片
    # -----------------------------------------------------

    for idx, image_name in enumerate(image_names):

        image_path = data_dir / image_name

        original = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )


        # 固定随机噪声
        rng_grid = np.random.default_rng(
            seed=42 + idx + noise_sigma * 100
        )


        noise = rng_grid.normal(
            loc=0,
            scale=noise_sigma,
            size=original.shape
        )


        noisy = (
            original.astype(np.float32)
            + noise
        )


        noisy = np.clip(
            noisy,
            0,
            255
        ).astype(np.uint8)



        # -------------------------------------------------
        # 搜索 sigmaColor × sigmaSpace
        # -------------------------------------------------

        for sigma_color in sigma_color_values:

            for sigma_space in sigma_space_values:


                result = cv2.bilateralFilter(
                    noisy,
                    d=9,
                    sigmaColor=sigma_color,
                    sigmaSpace=sigma_space
                )


                psnr, ssim = evaluate(
                    original,
                    result
                )


                grid_search_results.append({

                    "NoiseSigma": noise_sigma,

                    "Image": image_name,

                    "sigmaColor": sigma_color,

                    "sigmaSpace": sigma_space,

                    "PSNR": psnr,

                    "SSIM": ssim
                })



# =========================================================
# 计算平均结果
# =========================================================


print("\n")
print("=" * 80)
print("各噪声水平下最佳参数")
print("=" * 80)



best_parameters = {}



for noise_sigma in search_config.keys():


    print("\n")
    print(
        f"噪声 sigma_n={noise_sigma}"
    )


    current_noise_results = [

        r for r in grid_search_results

        if r["NoiseSigma"] == noise_sigma

    ]


    combinations = []


    for sigma_color in search_config[noise_sigma]["sigmaColor"]:

        for sigma_space in search_config[noise_sigma]["sigmaSpace"]:


            current = [

                r for r in current_noise_results

                if r["sigmaColor"] == sigma_color

                and r["sigmaSpace"] == sigma_space

            ]


            mean_psnr = np.mean([

                r["PSNR"]

                for r in current

            ])


            mean_ssim = np.mean([

                r["SSIM"]

                for r in current

            ])



            combinations.append({

                "sigmaColor": sigma_color,

                "sigmaSpace": sigma_space,

                "PSNR": mean_psnr,

                "SSIM": mean_ssim

            })


            print(

                f"r={sigma_color:3d}, "

                f"s={sigma_space}: "

                f"PSNR={mean_psnr:.2f} dB, "

                f"SSIM={mean_ssim:.4f}"

            )



    # ---------------------------------------------
    # PSNR最高组合
    # ---------------------------------------------

    best_psnr = max(

        combinations,

        key=lambda x:x["PSNR"]

    )


    # ---------------------------------------------
    # SSIM最高组合
    # ---------------------------------------------

    best_ssim = max(

        combinations,

        key=lambda x:x["SSIM"]

    )



    print("\nPSNR最佳：")

    print(

        f"r={best_psnr['sigmaColor']}, "

        f"s={best_psnr['sigmaSpace']}"

        f"   "

        f"PSNR={best_psnr['PSNR']:.2f}"

        f"   "

        f"SSIM={best_psnr['SSIM']:.4f}"

    )


    print("SSIM最佳：")


    print(

        f"r={best_ssim['sigmaColor']}, "

        f"s={best_ssim['sigmaSpace']}"

        f"   "

        f"PSNR={best_ssim['PSNR']:.2f}"

        f"   "

        f"SSIM={best_ssim['SSIM']:.4f}"

    )



    best_parameters[noise_sigma] = best_psnr



# =========================================================
# 保存全部搜索结果
# =========================================================


grid_csv = (

    result_dir

    / "bilateral_grid_search_results.csv"

)


with open(

    grid_csv,

    "w",

    newline="",

    encoding="utf-8-sig"

) as f:


    writer = csv.DictWriter(

        f,

        fieldnames=[

            "NoiseSigma",

            "Image",

            "sigmaColor",

            "sigmaSpace",

            "PSNR",

            "SSIM"

        ]

    )


    writer.writeheader()

    writer.writerows(

        grid_search_results

    )



print("\n二维搜索完成")
print(
    f"结果保存至：{grid_csv}"
)