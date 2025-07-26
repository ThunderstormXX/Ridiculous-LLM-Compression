import json
import matplotlib.pyplot as plt
from PIL import Image
import os

def plot_perplexity_log(log_path, output_img_path="perplexity_plot.jpg"):
    with open(log_path, 'r') as f:
        log = json.load(f)

    iterations = []
    pre_train_ppls = []
    post_train_ppls = []

    for entry in log:
        iterations.append(entry["iteration"])
        pre_train_ppls.append(entry.get("pre_train_perplexity"))
        post_train_ppls.append(entry.get("post_train_perplexity"))

    plt.figure(figsize=(12, 6))

    # === Отдельно рисуем точки ===
    for i in range(len(iterations)):
        if pre_train_ppls[i] is not None:
            plt.plot(iterations[i], pre_train_ppls[i], 'o', color='orange', label='Pre-train PPL' if i == 0 else "")
        if post_train_ppls[i] is not None:
            plt.plot(iterations[i], post_train_ppls[i], 'o', color='blue', label='Post-train PPL' if i == 0 else "")

    # === Стрелки ===
    for i in range(len(iterations)):
        # 🔵 Синяя: pre -> post
        if pre_train_ppls[i] is not None and post_train_ppls[i] is not None:
            plt.annotate(
                "", 
                xy=(iterations[i], post_train_ppls[i]), 
                xytext=(iterations[i], pre_train_ppls[i]),
                arrowprops=dict(arrowstyle="->", color="blue", lw=2)
            )
            mid_y = (pre_train_ppls[i] + post_train_ppls[i]) / 2
            plt.text(iterations[i] + 0.1, mid_y, "-- 500 steps", color="blue", fontsize=9, va="center")

        # 🔴 Красная: post[i-1] -> pre[i]
        if i > 0 and post_train_ppls[i - 1] is not None and pre_train_ppls[i] is not None:
            plt.annotate(
                "", 
                xy=(iterations[i], pre_train_ppls[i]), 
                xytext=(iterations[i - 1], post_train_ppls[i - 1]),
                arrowprops=dict(arrowstyle="->", color="red", lw=2)
            )
            mid_x = (iterations[i] + iterations[i - 1]) / 2
            mid_y = (pre_train_ppls[i] + post_train_ppls[i - 1]) / 2
            plt.text(mid_x, mid_y + 0.2, "-100 M", color="red", fontsize=9, ha="center")

    # 🔴 От итерации 0 к итерации 1 (если она есть)
    if pre_train_ppls[0] is not None and len(pre_train_ppls) > 1 and pre_train_ppls[1] is not None:
        plt.annotate(
            "", 
            xy=(iterations[1], pre_train_ppls[1]), 
            xytext=(iterations[0], pre_train_ppls[0]),
            arrowprops=dict(arrowstyle="->", color="red", lw=2)
        )
        mid_x = (iterations[0] + iterations[1]) / 2
        mid_y = (pre_train_ppls[0] + pre_train_ppls[1]) / 2
        plt.text(mid_x, mid_y + 0.2, "-100 M", color="red", fontsize=9, ha="center")

    plt.xlabel('Iteration')
    plt.ylabel('Perplexity')
    plt.title('Perplexity Jumps Across Iterations')
    plt.grid(True)
    plt.legend()
    plt.xticks(iterations)
    plt.tight_layout()

    # PNG -> JPEG
    tmp_path = "temp_plot.png"
    plt.savefig(tmp_path, dpi=300)
    plt.close()

    img = Image.open(tmp_path).convert("RGB")
    img.save(output_img_path, "JPEG", quality=95)
    os.remove(tmp_path)

    print(f"✅ JPEG plot saved as: {output_img_path}")

plot_perplexity_log("log.json", "perplexity_plot.jpg")
