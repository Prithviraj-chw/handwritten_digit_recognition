import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
import random
from PIL import Image, ImageOps
import io
import os
import time

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="✏️",
    layout="wide",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background: #0f0f1a; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #16162a;
        border-right: 1px solid #2a2a4a;
    }

    /* Cards */
    .stat-card {
        background: linear-gradient(135deg, #1e1e3a, #2a2a4a);
        border: 1px solid #3a3a6a;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .stat-card h2 { color: #7c7cff; font-size: 2rem; margin: 0; }
    .stat-card p  { color: #aaa; margin: 4px 0 0; font-size: 0.85rem; }

    /* Prediction badge */
    .pred-badge {
        background: linear-gradient(135deg, #7c7cff, #a855f7);
        color: white;
        font-size: 4rem;
        font-weight: bold;
        border-radius: 16px;
        padding: 20px 40px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 0 30px rgba(124,124,255,0.4);
    }
    .conf-text { color: #a0a0c0; font-size: 1.1rem; }

    /* Section headers */
    .section-title {
        color: #7c7cff;
        font-size: 1.2rem;
        font-weight: 600;
        border-bottom: 1px solid #2a2a4a;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* Training log */
    .log-box {
        background: #0a0a14;
        border: 1px solid #2a2a4a;
        border-radius: 8px;
        padding: 12px;
        font-family: monospace;
        font-size: 0.8rem;
        color: #88ff88;
        height: 220px;
        overflow-y: auto;
    }

    /* Hide Streamlit default footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Model Definition ─────────────────────────────────────────────────────────
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.25)
        )
        self.fc1 = nn.Linear(32 * 7 * 7, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ─── Session State Init ───────────────────────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model = None
if "trained" not in st.session_state:
    st.session_state.trained = False
if "train_losses" not in st.session_state:
    st.session_state.train_losses = []
if "test_accuracies" not in st.session_state:
    st.session_state.test_accuracies = []
if "log_lines" not in st.session_state:
    st.session_state.log_lines = []
if "test_dataset" not in st.session_state:
    st.session_state.test_dataset = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/home/claude/mnist_digit_cnn.pth"

# ─── Data helpers ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = datasets.MNIST(root="/home/claude/data", train=True,  transform=transform, download=True)
    test_ds  = datasets.MNIST(root="/home/claude/data", train=False, transform=transform, download=True)
    return train_ds, test_ds

def load_or_init_model():
    model = DigitCNN().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        return model, True
    return model, False

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✏️ Digit Recognizer")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Home", "🎓 Train Model", "🔍 Predict", "📊 Sample Predictions"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(f"**Device:** `{DEVICE}`")

    if os.path.exists(MODEL_PATH):
        st.success("✅ Saved model found")
    else:
        st.warning("⚠️ No saved model")

    if st.session_state.trained or os.path.exists(MODEL_PATH):
        if st.button("🗑️ Reset Model", use_container_width=True):
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            st.session_state.model = None
            st.session_state.trained = False
            st.session_state.train_losses = []
            st.session_state.test_accuracies = []
            st.session_state.log_lines = []
            st.rerun()

# ─── Page: Home ───────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("# Handwritten Digit Recognition")
    st.markdown("A CNN trained on MNIST — recognizes digits 0–9 from images you upload.")

    c1, c2, c3 = st.columns(3)
    with c1:
        acc_val = f"{st.session_state.test_accuracies[-1]:.1f}%" if st.session_state.test_accuracies else "—"
        st.markdown(f"""<div class="stat-card"><h2>{acc_val}</h2><p>Best Test Accuracy</p></div>""", unsafe_allow_html=True)
    with c2:
        ep_val = len(st.session_state.train_losses) if st.session_state.train_losses else "—"
        st.markdown(f"""<div class="stat-card"><h2>{ep_val}</h2><p>Epochs Trained</p></div>""", unsafe_allow_html=True)
    with c3:
        loss_val = f"{st.session_state.train_losses[-1]:.4f}" if st.session_state.train_losses else "—"
        st.markdown(f"""<div class="stat-card"><h2>{loss_val}</h2><p>Last Train Loss</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Architecture")
        st.markdown("""
| Layer | Details |
|-------|---------|
| Conv1 | 1→16 ch, 3×3, ReLU, MaxPool |
| Conv2 | 16→32 ch, 3×3, ReLU, MaxPool, Dropout |
| FC1   | 1568 → 100, ReLU |
| FC2   | 100 → 10 (logits) |
""")

    with col2:
        st.markdown("### Quick Start")
        st.markdown("""
1. **Train** — go to *Train Model* and click **Start Training**
2. **Predict** — go to *Predict* and upload a digit image
3. **Explore** — see *Sample Predictions* for MNIST test examples
        """)

# ─── Page: Train ──────────────────────────────────────────────────────────────
elif page == "🎓 Train Model":
    st.markdown("# 🎓 Train the CNN")

    col_cfg, col_log = st.columns([1, 2])

    with col_cfg:
        st.markdown('<div class="section-title">Hyperparameters</div>', unsafe_allow_html=True)
        epochs    = st.slider("Epochs", 1, 20, 5)
        lr        = st.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3], value=1e-3,
                                     format_func=lambda x: f"{x:.0e}")
        batch_sz  = st.select_slider("Batch Size", [64, 128, 256], value=128)
        run_btn   = st.button("🚀 Start Training", use_container_width=True, type="primary")

        if os.path.exists(MODEL_PATH):
            st.info("A saved model exists. Training will overwrite it.")

    with col_log:
        st.markdown('<div class="section-title">Training Log</div>', unsafe_allow_html=True)
        log_ph   = st.empty()
        prog_ph  = st.empty()
        chart_ph = st.empty()

        def render_log():
            html = "<br>".join(st.session_state.log_lines[-30:])
            log_ph.markdown(f'<div class="log-box">{html}</div>', unsafe_allow_html=True)

        render_log()

    if run_btn:
        st.session_state.train_losses     = []
        st.session_state.test_accuracies  = []
        st.session_state.log_lines        = ["Starting training…"]
        render_log()

        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
        ])
        train_ds, test_ds = get_datasets()
        train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=batch_sz, shuffle=False)

        model     = DigitCNN().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0
            for step, (imgs, lbls) in enumerate(train_loader):
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(imgs), lbls)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                if (step + 1) % 100 == 0:
                    msg = f"Ep {epoch}/{epochs} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}"
                    st.session_state.log_lines.append(msg)
                    render_log()

            avg_loss = epoch_loss / len(train_loader)

            # Evaluate
            model.eval(); correct = 0; t_loss = 0
            with torch.no_grad():
                for imgs, lbls in test_loader:
                    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                    out = model(imgs)
                    t_loss += criterion(out, lbls).item()
                    correct += (out.argmax(1) == lbls).sum().item()
            acc = 100 * correct / len(test_ds)

            st.session_state.train_losses.append(avg_loss)
            st.session_state.test_accuracies.append(acc)

            summary = f"✅ Epoch {epoch} done — AvgLoss: {avg_loss:.4f} | TestAcc: {acc:.2f}%"
            st.session_state.log_lines.append(summary)
            render_log()

            prog_ph.progress(epoch / epochs, text=f"Epoch {epoch}/{epochs}")

            # Live chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3), facecolor="#0f0f1a")
            for ax in (ax1, ax2):
                ax.set_facecolor("#0f0f1a")
                ax.tick_params(colors="#aaa"); ax.xaxis.label.set_color("#aaa")
                ax.yaxis.label.set_color("#aaa")
                for spine in ax.spines.values(): spine.set_edgecolor("#2a2a4a")

            ax1.plot(st.session_state.train_losses, color="#7c7cff", marker="o", linewidth=2)
            ax1.set_title("Train Loss", color="#ccc"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")

            ax2.plot(st.session_state.test_accuracies, color="#a855f7", marker="o", linewidth=2)
            ax2.set_title("Test Accuracy (%)", color="#ccc"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Acc %")
            ax2.set_ylim(0, 100)

            plt.tight_layout()
            chart_ph.pyplot(fig)
            plt.close(fig)

        torch.save(model.state_dict(), MODEL_PATH)
        st.session_state.model   = model
        st.session_state.trained = True
        st.session_state.log_lines.append("💾 Model saved to disk.")
        render_log()
        st.success(f"Training complete! Best accuracy: {max(st.session_state.test_accuracies):.2f}%")

# ─── Page: Predict ────────────────────────────────────────────────────────────
elif page == "🔍 Predict":
    st.markdown("# 🔍 Predict a Digit")

    model, loaded = load_or_init_model()
    if not loaded:
        st.warning("No saved model found. Please train first.")
        st.stop()

    model.eval()

    col_up, col_res = st.columns([1, 1])

    with col_up:
        st.markdown('<div class="section-title">Upload Image</div>', unsafe_allow_html=True)
        st.markdown("Upload any image of a handwritten digit (PNG/JPG). Works best with dark background and white digit, or vice versa.")
        uploaded = st.file_uploader("Choose an image…", type=["png", "jpg", "jpeg"])

        if uploaded:
            pil_img = Image.open(uploaded).convert("L")
            st.image(pil_img, caption="Uploaded image", use_container_width=True)

    with col_res:
        if uploaded:
            st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)

            # Preprocess: resize, invert if needed, normalize
            img_resized = pil_img.resize((28, 28))
            arr = np.array(img_resized, dtype=np.float32)

            # Auto-invert: MNIST has white digit on black background
            if arr.mean() > 128:
                img_resized = ImageOps.invert(img_resized)
                arr = np.array(img_resized, dtype=np.float32)

            tensor = torch.tensor(arr / 255.0).unsqueeze(0).unsqueeze(0)
            tensor = transforms.Normalize((0.5,), (0.5,))(tensor).to(DEVICE)

            with torch.no_grad():
                logits = model(tensor)
                probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                pred   = int(probs.argmax())
                conf   = float(probs.max()) * 100

            st.markdown(f'<div class="pred-badge">{pred}</div>', unsafe_allow_html=True)
            st.markdown(f'<p class="conf-text" style="text-align:center">Confidence: <b>{conf:.1f}%</b></p>', unsafe_allow_html=True)

            # Bar chart of all probabilities
            fig, ax = plt.subplots(figsize=(7, 3), facecolor="#0f0f1a")
            ax.set_facecolor("#0f0f1a")
            colors = ["#a855f7" if i == pred else "#3a3a6a" for i in range(10)]
            bars = ax.bar(range(10), probs * 100, color=colors, edgecolor="#2a2a4a")
            ax.set_xticks(range(10)); ax.set_xlabel("Digit", color="#aaa")
            ax.set_ylabel("Probability (%)", color="#aaa")
            ax.set_title("Class Probabilities", color="#ccc")
            ax.tick_params(colors="#aaa")
            for spine in ax.spines.values(): spine.set_edgecolor("#2a2a4a")
            ax.set_ylim(0, 100)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ─── Page: Sample Predictions ─────────────────────────────────────────────────
elif page == "📊 Sample Predictions":
    st.markdown("# 📊 Sample MNIST Predictions")

    model, loaded = load_or_init_model()
    if not loaded:
        st.warning("No saved model found. Please train first.")
        st.stop()

    model.eval()
    _, test_ds = get_datasets()

    n_samples = st.slider("Number of samples", 4, 24, 12, step=4)
    if st.button("🔀 Shuffle Samples", use_container_width=True) or True:
        indices = random.sample(range(len(test_ds)), n_samples)

    cols_per_row = 6
    rows = (n_samples + cols_per_row - 1) // cols_per_row

    for row in range(rows):
        cols = st.columns(cols_per_row)
        for col_i in range(cols_per_row):
            idx = row * cols_per_row + col_i
            if idx >= n_samples:
                break
            img_tensor, label = test_ds[indices[idx]]
            inp = img_tensor.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = model(inp).argmax(dim=1).item()

            fig, ax = plt.subplots(figsize=(2, 2), facecolor="#1e1e3a")
            ax.set_facecolor("#1e1e3a")
            ax.imshow(img_tensor.squeeze(), cmap="gray")
            color = "#88ff88" if pred == label else "#ff6666"
            ax.set_title(f"T:{label} P:{pred}", color=color, fontsize=9)
            ax.axis("off")
            plt.tight_layout(pad=0.3)
            cols[col_i].pyplot(fig)
            plt.close(fig)

    st.caption("🟢 Green = correct prediction  |  🔴 Red = incorrect prediction")