# qwen
diff --git a/README.md b/README.md
index 1549943df78c38a144c7acdc3f51134e3bb66d4f..b1cd08ba55e040400e57d0265324ee13462d844f 100644
--- a/README.md
+++ b/README.md
@@ -1 +1,17 @@
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
+输出包含基模型与微调模型的指标对比，用于查看微调带来的性能提升。
