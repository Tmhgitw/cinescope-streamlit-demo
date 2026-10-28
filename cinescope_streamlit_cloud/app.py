"""Local Streamlit demo for movie-genre prediction."""

from __future__ import annotations

import os

import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import nn
from torchvision import models, transforms
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


CLASSES = ("action", "comedy", "horror", "romance")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}
CHINESE_NAMES = {
    "action": "动作片",
    "comedy": "喜剧片",
    "horror": "恐怖片",
    "romance": "爱情片",
}
GENRE_COLORS = {
    "action": "#ff6b4a",
    "comedy": "#ffc857",
    "horror": "#c084fc",
    "romance": "#ff73a8",
}
GENRE_EMOJIS = {
    "action": "⚡",
    "comedy": "✨",
    "horror": "🌙",
    "romance": "♥",
}
DEFAULT_MODEL_REPO_ID = "YOUR_HF_USERNAME/cinescope-movie-genre-models"


def get_model_repo_id() -> str:
    """Read the public Hugging Face model repository from Streamlit settings."""
    environment_value = os.getenv("HF_MODEL_REPO_ID", "").strip()
    if environment_value:
        return environment_value
    try:
        secret_value = str(st.secrets["HF_MODEL_REPO_ID"]).strip()
        if secret_value:
            return secret_value
    except Exception:
        pass
    return DEFAULT_MODEL_REPO_ID


class CrossAttentionClassifier(nn.Module):
    """The same fusion architecture used by 04_train_multimodal.py."""

    def __init__(self, d_model: int = 256, num_classes: int = 4) -> None:
        super().__init__()
        self.text_projection = nn.Linear(768, d_model)
        self.image_projection = nn.Conv2d(2048, d_model, kernel_size=1)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Conv2d(d_model, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, text_features, attention_mask, image_features):
        text = self.text_projection(text_features)
        image_2d = self.image_projection(image_features)
        batch, channels, height, width = image_2d.shape
        image = image_2d.flatten(2).transpose(1, 2)
        attended_text, _ = self.cross_attention(
            query=image,
            key=text,
            value=text,
            key_padding_mask=~attention_mask.bool(),
            need_weights=False,
        )
        fused = self.fusion_norm(image + attended_text)
        fused_2d = fused.transpose(1, 2).reshape(batch, channels, height, width)
        return self.classifier(fused_2d)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource(show_spinner="正在加载模型，第一次可能需要一些时间……")
def load_models():
    device = select_device()
    model_repo_id = get_model_repo_id()
    if model_repo_id == DEFAULT_MODEL_REPO_ID:
        raise RuntimeError(
            "尚未设置 HF_MODEL_REPO_ID。请在 Streamlit 的 Advanced settings → "
            "Secrets 中填写你的 Hugging Face 模型仓库地址。"
        )

    image_checkpoint_path = hf_hub_download(
        repo_id=model_repo_id,
        filename="image_best_model.pt",
    )
    text_checkpoint_path = hf_hub_download(
        repo_id=model_repo_id,
        filename="text_best_model.pt",
    )
    multimodal_checkpoint_path = hf_hub_download(
        repo_id=model_repo_id,
        filename="multimodal_best_model.pt",
    )

    image_model = models.resnet50(weights=None)
    image_model.fc = nn.Linear(image_model.fc.in_features, len(CLASSES))
    image_checkpoint = torch.load(
        image_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    image_model.load_state_dict(image_checkpoint["model_state"])
    image_model = image_model.to(device).eval()
    image_encoder = nn.Sequential(*list(image_model.children())[:-2]).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text_config = AutoConfig.from_pretrained(
        "bert-base-uncased",
        num_labels=len(CLASSES),
        id2label={index: name for index, name in enumerate(CLASSES)},
        label2id=CLASS_TO_ID,
    )
    text_classifier = AutoModelForSequenceClassification.from_config(text_config)
    text_classifier.load_state_dict(
        torch.load(
            text_checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    )
    text_encoder = text_classifier.base_model.to(device).eval()

    fusion_model = CrossAttentionClassifier(num_classes=len(CLASSES))
    fusion_model.load_state_dict(
        torch.load(
            multimodal_checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    )
    fusion_model = fusion_model.to(device).eval()
    return device, image_model, image_encoder, tokenizer, text_encoder, fusion_model


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict_image(image: Image.Image, bundle) -> torch.Tensor:
    device, image_model, _, _, _, _ = bundle
    tensor = IMAGE_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = image_model(tensor)
    return torch.softmax(logits, dim=1).squeeze(0).cpu()


def predict_multimodal(image: Image.Image, description: str, bundle) -> torch.Tensor:
    device, _, image_encoder, tokenizer, text_encoder, fusion_model = bundle
    image_tensor = IMAGE_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    encoded = tokenizer(
        description,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        image_features = image_encoder(image_tensor)
        text_features = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        logits = fusion_model(text_features, attention_mask, image_features)
    return torch.softmax(logits, dim=1).squeeze(0).cpu()


def show_result(probabilities: torch.Tensor) -> None:
    best_index = int(probabilities.argmax())
    best_class = CLASSES[best_index]
    confidence = float(probabilities[best_index])
    st.markdown(
        f"""
        <div class="result-card" style="--accent:{GENRE_COLORS[best_class]}">
          <div class="result-eyebrow">PREDICTION</div>
          <div class="result-main">
            <span class="result-icon">{GENRE_EMOJIS[best_class]}</span>
            <div>
              <div class="result-label">{CHINESE_NAMES[best_class]}</div>
              <div class="result-en">{best_class.upper()}</div>
            </div>
            <div class="confidence"><strong>{confidence:.1%}</strong><span>置信度</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="prob-title">四种类型概率</div>', unsafe_allow_html=True)
    for name, value in sorted(
        zip(CLASSES, probabilities.tolist()), key=lambda item: item[1], reverse=True
    ):
        percentage = float(value) * 100
        st.markdown(
            f"""
            <div class="prob-row">
              <div class="prob-meta"><span>{GENRE_EMOJIS[name]} {CHINESE_NAMES[name]}</span><b>{percentage:.1f}%</b></div>
              <div class="prob-track"><div class="prob-fill" style="width:{percentage:.1f}%;background:{GENRE_COLORS[name]}"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption("置信度是模型在四个候选类别之间的相对判断，不代表电影只能属于一个类型。")


st.set_page_config(page_title="CineScope · 电影类型识别", page_icon="🎬", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Noto+Sans+SC:wght@400;500;600;700&display=swap');
    :root { --ink:#f7f3ea; --muted:#a8a4b3; --panel:rgba(25,23,34,.78); --line:rgba(255,255,255,.10); }
    .stApp {
      background:
        radial-gradient(circle at 13% 12%, rgba(140,73,255,.18), transparent 30%),
        radial-gradient(circle at 87% 18%, rgba(255,91,105,.14), transparent 27%),
        linear-gradient(145deg, #0a0910 0%, #12101a 52%, #08070c 100%);
      color: var(--ink);
    }
    .block-container { max-width: 1120px; padding-top: 2.4rem; padding-bottom: 4rem; }
    [data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }
    html, body, [class*="css"] { font-family: "DM Sans", "Noto Sans SC", sans-serif; }
    .hero {
      padding: 3.2rem 3rem 2.8rem; border: 1px solid var(--line); border-radius: 28px;
      background: linear-gradient(120deg, rgba(30,27,42,.88), rgba(18,16,27,.62));
      box-shadow: 0 32px 90px rgba(0,0,0,.32); position: relative; overflow: hidden;
    }
    .hero:after {
      content:""; position:absolute; width:280px; height:280px; right:-70px; top:-100px;
      border:1px solid rgba(255,255,255,.08); border-radius:50%; box-shadow:0 0 0 38px rgba(255,255,255,.025),0 0 0 78px rgba(255,255,255,.018);
    }
    .kicker { color:#c4a7ff; letter-spacing:.18em; font-size:.76rem; font-weight:700; margin-bottom:1rem; }
    .hero h1 { font-size:clamp(2.25rem,5vw,4.2rem); line-height:1.03; margin:0; letter-spacing:-.055em; color:#fffaf0; }
    .hero p { max-width:650px; color:#bcb7c7; font-size:1.05rem; margin:1.2rem 0 1.5rem; line-height:1.75; }
    .badges { display:flex; flex-wrap:wrap; gap:.65rem; }
    .badge { padding:.48rem .78rem; border:1px solid var(--line); border-radius:999px; color:#ded8e8; background:rgba(255,255,255,.035); font-size:.82rem; }
    .section-head { margin:2.5rem 0 1rem; }
    .section-head b { display:block; font-size:1.35rem; color:#fff; }
    .section-head span { color:var(--muted); font-size:.92rem; }
    [data-testid="stFileUploader"] { padding:1rem; border:1px solid var(--line); border-radius:18px; background:var(--panel); }
    [data-testid="stFileUploaderDropzone"] {
      min-height:150px; padding:1.4rem 1.2rem; border:1px dashed rgba(196,167,255,.42);
      border-radius:15px; background:linear-gradient(135deg,rgba(143,92,255,.09),rgba(232,93,134,.045));
      transition:border-color .2s ease, background .2s ease, transform .2s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
      border-color:rgba(196,167,255,.78); background:linear-gradient(135deg,rgba(143,92,255,.15),rgba(232,93,134,.07));
      transform:translateY(-1px);
    }
    [data-testid="stFileUploaderDropzone"] svg { color:#b995ff !important; fill:#b995ff !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] span { color:#eee8f6 !important; font-weight:650; }
    [data-testid="stFileUploaderDropzoneInstructions"] small { color:#8f899d !important; }
    [data-testid="stFileUploaderDropzone"] button {
      min-height:auto; padding:.58rem .9rem; border:1px solid rgba(196,167,255,.3) !important;
      border-radius:10px; background:rgba(196,167,255,.10) !important; color:#eee8f6 !important;
      box-shadow:none !important; font-weight:650;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
      border-color:rgba(196,167,255,.6) !important; background:rgba(196,167,255,.17) !important;
    }
    [data-testid="stFileUploaderFile"] { border:1px solid var(--line); border-radius:12px; background:rgba(255,255,255,.035); }
    [data-testid="stFileUploaderFile"] * { color:#dcd6e4 !important; }
    [data-testid="stWidgetLabel"] p { color:#c7c1cf; font-weight:600; }
    [data-testid="stImage"] img { border-radius:18px; box-shadow:0 20px 55px rgba(0,0,0,.4); }
    [data-testid="stTextArea"] textarea { background:rgba(10,9,16,.8); border:1px solid var(--line); border-radius:14px; color:#f6f1e8; }
    div[role="radiogroup"] { padding:.4rem; border:1px solid var(--line); border-radius:16px; background:var(--panel); }
    [data-testid="stRadio"] > label p { color:#e9e4ef !important; font-weight:700 !important; }
    [data-testid="stRadio"] label[data-baseweb="radio"] { padding:.35rem .45rem; border-radius:10px; }
    [data-testid="stRadio"] label[data-baseweb="radio"] p,
    [data-testid="stRadio"] label[data-baseweb="radio"] span {
      color:#d7d1df !important; opacity:1 !important; font-weight:600 !important;
    }
    [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
      background:rgba(196,167,255,.10);
    }
    [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p,
    [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) span {
      color:#fffaf2 !important; font-weight:750 !important;
    }
    .mode-note { padding:1rem 1.1rem; border-radius:14px; background:rgba(196,167,255,.08); border:1px solid rgba(196,167,255,.18); color:#cfc8da; margin:.4rem 0 1rem; font-size:.9rem; }
    .stButton > button { min-height:3.25rem; border-radius:14px; font-weight:700; border:0; background:linear-gradient(100deg,#8f5cff,#e85d86); color:white; box-shadow:0 12px 28px rgba(143,92,255,.22); }
    .stButton > button:hover { color:white; border:0; transform:translateY(-1px); }
    .result-card { margin-top:1.3rem; padding:1.4rem 1.5rem; border-radius:18px; border:1px solid color-mix(in srgb, var(--accent) 38%, transparent); background:linear-gradient(120deg,color-mix(in srgb,var(--accent) 13%, #17141f),rgba(23,20,31,.82)); }
    .result-eyebrow { font-size:.68rem; letter-spacing:.17em; color:var(--accent); font-weight:700; margin-bottom:.8rem; }
    .result-main { display:flex; align-items:center; gap:1rem; }
    .result-icon { display:grid; place-items:center; width:46px; height:46px; border-radius:14px; background:color-mix(in srgb,var(--accent) 18%, transparent); font-size:1.3rem; }
    .result-label { color:#fff; font-size:1.55rem; font-weight:700; line-height:1.05; }
    .result-en { color:#9892a3; font-size:.68rem; letter-spacing:.14em; margin-top:.3rem; }
    .confidence { margin-left:auto; text-align:right; }
    .confidence strong { display:block; color:#fff; font-size:1.55rem; }
    .confidence span { color:#9892a3; font-size:.75rem; }
    .prob-title { margin:1.4rem 0 .8rem; color:#f5f0e7; font-weight:700; }
    .prob-row { margin:.75rem 0; }
    .prob-meta { display:flex; justify-content:space-between; color:#bcb7c7; font-size:.86rem; margin-bottom:.35rem; }
    .prob-meta b { color:#eee9f4; }
    .prob-track { height:8px; border-radius:999px; overflow:hidden; background:rgba(255,255,255,.07); }
    .prob-fill { height:100%; border-radius:999px; box-shadow:0 0 15px rgba(255,255,255,.10); }
    @media (max-width:700px) { .hero { padding:2.2rem 1.4rem; } .block-container { padding:1rem; } }
    </style>
    <section class="hero">
      <div class="kicker">CINESCOPE · LOCAL AI LAB</div>
      <h1>从一张海报，<br>读懂电影的气质。</h1>
      <p>上传电影海报，并可选填英文剧情简介。模型会结合视觉线索与文本语义，判断最可能的电影类型。</p>
      <div class="badges"><span class="badge">4 种电影类型</span><span class="badge">多模态准确率 82.44%</span><span class="badge">云端交互 Demo</span></div>
    </section>
    <div class="section-head"><b>开始一次预测</b><span>选择模式，然后提供电影信息</span></div>
    """,
    unsafe_allow_html=True,
)

mode = st.radio(
    "预测模式",
    ("快速模式：只看海报", "精确模式：海报＋英文简介"),
    horizontal=True,
)

left, right = st.columns([0.9, 1.1], gap="large")
with left:
    uploaded_file = st.file_uploader("01 · 上传电影海报", type=["jpg", "jpeg", "png"])
    image = None
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="海报预览", use_container_width=True)
        except Exception:
            st.error("无法读取这张图片，请换一张 JPG 或 PNG 图片。")

with right:
    description = ""
    if mode.startswith("精确模式"):
        st.markdown('<div class="mode-note">精确模式使用海报和剧情简介，测试准确率为 82.44%。简介请使用英文。</div>', unsafe_allow_html=True)
        description = st.text_area(
            "02 · 输入英文电影简介",
            height=155,
            placeholder="Example: A detective investigates a series of mysterious disappearances...",
        )
    else:
        st.markdown('<div class="mode-note">快速模式只分析海报视觉特征，测试准确率为 74.81%。</div>', unsafe_allow_html=True)

    if st.button("分析这部电影 →", type="primary", use_container_width=True):
        if image is None:
            st.warning("请先上传电影海报。")
        elif mode.startswith("精确模式") and not description.strip():
            st.warning("精确模式还需要填写英文电影简介。")
        else:
            try:
                bundle = load_models()
                with st.spinner("模型正在分析海报与剧情……"):
                    if mode.startswith("快速模式"):
                        probabilities = predict_image(image, bundle)
                    else:
                        probabilities = predict_multimodal(image, description.strip(), bundle)
                show_result(probabilities)
            except Exception as error:
                st.error(f"预测失败：{error}")
                st.info("请确认模型仓库为 Public，且仓库根目录中存在三个指定名称的 .pt 文件。")
