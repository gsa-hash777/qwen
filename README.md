diff --git a/README.md b/README.md
index 1549943df78c38a144c7acdc3f51134e3bb66d4f..b94d98d2f2e7f07bc0d24291671051ed1ea9693d 100644
--- a/README.md
+++ b/README.md
@@ -1 +1,25 @@
-# qwen
\ No newline at end of file
+# qwen
+
+## LoRA 微调前后评测
+
+仓库内提供 `qwen/eval.py`，可基于 `val2014.csv` 对 **基模型** 与 **LoRA 微调模型** 做统一评测，输出常见指标（BLEU、ROUGE-L、METEOR，可用时包含 CIDEr）。
+
+```bash
+cd qwen
+python eval.py \
+  --base-model ./Qwen/Qwen2.5-VL-7B-Instruct \
+  --lora-path ./output/Qwen2_5-VL-7B/checkpoint-930 \
+  --val-csv val2014.csv \
+  --max-samples 200 \
+  --output eval_metrics.json
+```
+
+如遇到 `FileNotFoundError: Module 'sacrebleu' doesn't exist` 的报错，说明评测指标依赖未安装，
+请先安装评测依赖：
+
+```bash
+pip install evaluate sacrebleu rouge-score nltk
+python -m nltk.downloader wordnet
+```
+
+输出包含基模型与微调模型的指标对比，用于查看微调带来的性能提升。
