# CineScope：电影类型多模态识别 Demo

上传电影海报，并可选填英文剧情简介。网页会调用训练好的 ResNet-50、BERT 和跨模态注意力模型，预测动作、喜剧、恐怖、爱情四种电影类型。

## 测试结果

| 模型 | 输入 | 测试准确率 | Macro F1 |
|---|---|---:|---:|
| ResNet-50 | 海报 | 74.81% | 71.70% |
| BERT | 英文简介 | 71.76% | 61.85% |
| 多模态融合 | 海报 + 英文简介 | 82.44% | 79.68% |

## 文件说明

- `app.py`：Streamlit 网页与推理代码
- `requirements.txt`：部署时需要安装的 Python 库
- 模型权重不保存在 GitHub；应用首次预测时会从 Hugging Face Model 仓库下载并缓存

## Streamlit Cloud 设置

部署时把入口文件设为 `app.py`。在 **Advanced settings → Secrets** 中填写：

```toml
HF_MODEL_REPO_ID = "你的HuggingFace用户名/cinescope-movie-genre-models"
```

Hugging Face 模型仓库必须包含以下三个文件：

```text
image_best_model.pt
text_best_model.pt
multimodal_best_model.pt
```

模型仓库设为 Public 时不需要填写 Hugging Face Token。
