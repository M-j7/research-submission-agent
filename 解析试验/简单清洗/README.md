# IEEE Access 简单规则清洗

本轮只生成派生清洗稿，原始资料与 SimpleDirectoryReader 试读稿不覆盖。

- 当前应查看的结果：[2026-10-06 final](结果/20261006-final/README.md)
- 清洗实现：[clean_trial.py](scripts/clean_trial.py)
- 回归测试：[test_clean_trial.py](tests/test_clean_trial.py)
- `结果/20261006-first-pass`、`结果/20261006-second-pass` 和 `结果/20261006-reviewed-pass` 是规则边界修订过程中的旧试验稿，请勿作为当前结果；使用 `final`。

本轮范围：IEEE Access 当前原始资料库中的 22 份 HTML、8 份 PDF、1 份 DOCX。已删除的 7 份网页打印 PDF 不在输入范围。18 份文件从官方来源清单带入来源 URL 与采集时间；另外 13 份将缺失信息标为未记录，没有用文件时间补值。DOCX 只做空白规范化，模板示例与占位内容保留并标记为模板。
