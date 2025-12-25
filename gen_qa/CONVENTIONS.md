# QA-Extract 项目规约

## Chunk分块部分
提取问答对之前需要对原文档进行分块，无论是md文档或者是其他格式的原文本都需要按照下面要求的分块输出格式进行输出
### 分块输出格式（JSON）
- 存储形式：单个 JSON 文件，整体为列表。
- 列表元素：每个元素表示一个分块的 JSON 对象，需包含以下字段：
  - `chunk_id`：分块的 UUID（字符串）。
  - `chunk_content`：分块的原始文本内容。
  - `chunk_len`：分块的字符数（建议按 `len(chunk_content)` 计算）。
  - `header`：分块所在的标题路径，多个级别的标题按出现顺序用空格拼接成单个字符串。

示例（节选）：
```json
[
  {
    "chunk_id": "8f3c2a9e-6b0b-4b9c-9c1e-5c2f9e5f7a31",
    "chunk_content": "……分块内容……",
    "chunk_len": 1234,
    "header": "Header1 Header2 Header3"
  }
]
```
