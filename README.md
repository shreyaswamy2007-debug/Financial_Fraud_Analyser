# Financial_Fraud_Analyser
"""FraudShield AI Pro - hackathon version.

Install:  pip install gradio pandas numpy scikit-learn
Run:      python3 app_pro.py
NOTE: The ML model is trained on synthetic data (demo only).
"""
import copy
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from datetime import datetime

import gradio as gr
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from streamlit import tabs

# =====================================================================
# 1. SCAM DATABASE (saved to a JSON file so reports persist)
# =====================================================================
DB_FILE = "reported_upi.json"
DEFAULT_REPORTED = {
    "winner.prize@upi": {"count": 14, "type": "Lottery scam"},
    "kyc.update@paytm": {"count": 9, "type": "Fake KYC"},
}
VERIFIED = {"amazon@apl", "swiggy@icici", "zomato@icici", "irctc@sbi"}


def load_db():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return copy.deepcopy(DEFAULT_REPORTED)


def save_db():
    with open(DB_FILE, "w") as f:
        json.dump(REPORTED, f, indent=2)


REPORTED = load_db()

# =====================================================================
# 2. SMS SCANNER (English + Hinglish, link and brand analysis)
# =====================================================================
BRANDS = {
    "sbi": "sbi.co.in", "hdfc": "hdfcbank.com", "icici": "icicibank.com",
    "paytm": "paytm.com", "phonepe": "phonepe.com", "amazon": "amazon.in",
}
SMS_RULES = [
    (r"https?://|www\.", 20, "Contains a link"),
    (r"bit\.ly|tinyurl|t\.co|goo\.gl|cutt\.ly|rb\.gy|is\.gd", 20, "Shortened link hides the real site"),
    (r"\botp\b|\bpin\b|\bcvv\b", 25, "Asks for OTP/PIN/CVV"),
    (r"blocked|suspended|kyc|expire|deactivat|band ho jayega|block ho jayega", 25, "Fake bank alert wording"),
    (r"lottery|winner|you won|won a|prize|lucky draw|reward", 25, "Lottery/prize bait"),
    (r"cashback|refund|free gift|congratulations", 15, "Too-good-to-be-true offer"),
    (r"urgent|immediately|within 24|act now|last chance|turant|abhi click", 15, "Urgency pressure"),
    (r"processing fee|send money|transfer|pay now", 20, "Asks for money"),
    (r"\.apk|install the app|anydesk|teamviewer|quicksupport", 30, "Asks to install a remote-access/unknown app"),
]


def sms_scam_check(text):
    t = text.lower()
    score, reasons = 0, []
    for pattern, pts, why in SMS_RULES:
        if re.search(pattern, t):
            score += pts
            reasons.append(why)

    urls = re.findall(r"(?:https?://|www\.)[^\s]+", t)
    domains = [re.sub(r"^https?://|^www\.", "", u).split("/")[0] for u in urls]
    for brand, official in BRANDS.items():
        if re.search(rf"\b{brand}\b", t) and domains and not any(d.endswith(official) for d in domains):
            score += 30
            reasons.append(f"Mentions {brand.upper()} but link is not {official}")
            break
    if any(re.search(r"\.(xyz|top|click|loan|buzz|icu|cfd)$", d) for d in domains):
        score += 15
        reasons.append("Link uses a suspicious domain ending")

    score = min(score, 100)
    level = "HIGH" if score >= 60 else "MEDIUM" if score >= 30 else "LOW"
    return {"risk_score": score, "level": level, "reasons": reasons, "urls": urls}


# =====================================================================
# 3. UPI ID CHECKER
# =====================================================================
SUSPICIOUS_WORDS = r"prize|lottery|winner|kyc|refund|reward|support|helpdesk|cashback|loan|claim"


def upi_check(upi_id):
    u = upi_id.strip().lower()
    if not re.fullmatch(r"[a-z0-9.\-_]{2,}@[a-z]{2,}", u):
        return {"status": "INVALID", "risk": 25, "msg": "Not a valid UPI ID format"}
    if u in REPORTED:
        r = REPORTED[u]
        return {"status": "REPORTED", "risk": 90, "msg": f"Reported {r['count']} times: {r['type']}"}
    if u in VERIFIED:
        return {"status": "VERIFIED", "risk": 0, "msg": "Known verified merchant"}
    if re.search(SUSPICIOUS_WORDS, u.split("@")[0]):
        return {"status": "SUSPICIOUS", "risk": 45, "msg": "ID name contains scam-style words"}
    return {"status": "UNVERIFIED", "risk": 15, "msg": "Not in the database. Verify before paying."}


# =====================================================================
# 4. TRANSACTION ML MODEL (synthetic data) + explainability
# =====================================================================
FEATURES = ["amount", "hour", "avg_spend_30d", "txns_last_24h",
            "new_device", "new_receiver", "amount_ratio"]
PRETTY = {
    "amount": "Payment amount", "hour": "Time of day",
    "txns_last_24h": "Number of recent transactions",
    "new_device": "New device", "new_receiver": "New receiver",
}

rng = np.random.default_rng(42)
N = 60000
df = pd.DataFrame({
    "amount": rng.exponential(80, N),
    "hour": rng.integers(0, 24, N),
    "avg_spend_30d": rng.normal(80, 20, N).clip(5),
    "txns_last_24h": rng.poisson(3, N),
    "new_device": rng.binomial(1, 0.05, N),
    "new_receiver": rng.binomial(1, 0.30, N),
})
df["amount_ratio"] = df.amount / df.avg_spend_30d
rule = (
    ((df.amount_ratio > 4) & (df.new_device == 1))
    | ((df.amount > 500) & (df.hour < 4))
    | (df.txns_last_24h > 10)
    | ((df.amount_ratio > 6) & (df.new_receiver == 1) & (df.hour < 5))
)
flip = rng.random(N) < 0.02  # 2% label noise so the model is not a perfect copy of the rules
df["is_fraud"] = (rule ^ flip).astype(int)

X, y = df[FEATURES], df["is_fraud"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = RandomForestClassifier(n_estimators=150, class_weight="balanced",
                               random_state=42, n_jobs=-1)
model.fit(X_tr, y_tr)
_pred = model.predict(X_te)
_proba = model.predict_proba(X_te)[:, 1]
METRICS = {
    "Precision": precision_score(y_te, _pred),
    "Recall": recall_score(y_te, _pred),
    "F1 score": f1_score(y_te, _pred),
    "ROC-AUC": roc_auc_score(y_te, _proba),
}
IMPORTANCES = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])


def _prob(row):
    return float(model.predict_proba(pd.DataFrame([row])[FEATURES])[0][1])


def transaction_check(t):
    row = {**t, "amount_ratio": t["amount"] / max(t["avg_spend_30d"], 1)}
    prob = _prob(row)

    # Explainability: how much does the risk drop if each factor looked normal?
    baseline = {"amount": t["avg_spend_30d"], "hour": 14, "txns_last_24h": 3,
                "new_device": 0, "new_receiver": 0}
    drivers = []
    for f, normal in baseline.items():
        alt = {**row, f: normal}
        alt["amount_ratio"] = alt["amount"] / max(alt["avg_spend_30d"], 1)
        drop = prob - _prob(alt)
        if drop > 0.02:
            drivers.append((PRETTY[f], drop))
    drivers.sort(key=lambda x: -x[1])

    reasons = []
    ratio = row["amount_ratio"]
    if ratio > 4:
        reasons.append(f"Amount is {ratio:.1f}x your 30-day average")
    if t["txns_last_24h"] > 10:
        reasons.append("Unusually many transactions in 24 hours")
    if t["new_device"]:
        reasons.append("Payment from a new device")
    if t["new_receiver"]:
        reasons.append("Paying a first-time receiver")
    if t["hour"] < 4 and t["amount"] > 300:
        reasons.append("Large payment late at night")
    return prob, reasons, drivers


# =====================================================================
# 5. COMBINED RISK ENGINE
# =====================================================================
STEPS = {
    "English": {
        "HIGH": ["Do NOT proceed with this payment", "Never share your OTP or PIN",
                 "Call your bank's official number",
                 "Report fraud: call 1930 or visit cybercrime.gov.in"],
        "MEDIUM": ["Verify the receiver first", "Check the amount carefully"],
        "LOW": ["Looks normal. Stay alert."],
    },
    "Hindi (हिंदी)": {
        "HIGH": ["यह भुगतान न करें", "अपना OTP या PIN कभी साझा न करें",
                 "अपने बैंक के आधिकारिक नंबर पर कॉल करें",
                 "धोखाधड़ी की रिपोर्ट करें: 1930 पर कॉल करें या cybercrime.gov.in पर जाएँ"],
        "MEDIUM": ["पहले प्राप्तकर्ता की जाँच करें", "राशि ध्यान से जाँचें"],
        "LOW": ["सब सामान्य लगता है। सतर्क रहें।"],
    },
    "Kannada (ಕನ್ನಡ)": {
        "HIGH": ["ಈ ಪಾವತಿಯನ್ನು ಮುಂದುವರಿಸಬೇಡಿ",
                 "ನಿಮ್ಮ OTP ಅಥವಾ PIN ಅನ್ನು ಯಾರೊಂದಿಗೂ ಹಂಚಿಕೊಳ್ಳಬೇಡಿ",
                 "ನಿಮ್ಮ ಬ್ಯಾಂಕ್‌ನ ಅಧಿಕೃತ ಸಂಖ್ಯೆಗೆ ಕರೆ ಮಾಡಿ",
                 "ವಂಚನೆಯನ್ನು ವರದಿ ಮಾಡಿ: 1930 ಗೆ ಕರೆ ಮಾಡಿ ಅಥವಾ cybercrime.gov.in ಗೆ ಭೇಟಿ ನೀಡಿ"],
        "MEDIUM": ["ಮೊದಲು ಸ್ವೀಕರಿಸುವವರನ್ನು ಪರಿಶೀಲಿಸಿ",
                   "ಮೊತ್ತವನ್ನು ಎಚ್ಚರಿಕೆಯಿಂದ ಪರಿಶೀಲಿಸಿ"],
        "LOW": ["ಎಲ್ಲವೂ ಸಾಮಾನ್ಯವಾಗಿ ಕಾಣುತ್ತಿದೆ. ಎಚ್ಚರಿಕೆಯಿಂದಿರಿ."],
    },
}


def level_of(score):
    return "HIGH" if score >= 60 else "MEDIUM" if score >= 30 else "LOW"


def combine(txn, upi, sms, lang):
    prob, reasons, drivers = transaction_check(txn)
    score = prob * 100
    flags = 1 if prob >= 0.5 else 0

    if upi and upi.strip():
        info = upi_check(upi)
        if info["risk"] >= 45:
            score = max(score, info["risk"])
            flags += 1
        else:
            score = max(0, score + info["risk"] - (10 if info["status"] == "VERIFIED" else 0))
        if info["status"] != "VERIFIED":
            reasons.append("UPI: " + info["msg"])

    if sms and sms.strip():
        s = sms_scam_check(sms)
        if s["level"] != "LOW":
            score = max(score, s["risk_score"])
            flags += 1
            reasons += ["SMS: " + r for r in s["reasons"]]

    if flags >= 2:
        score += 10
        reasons.append("Multiple independent red flags together")

    score = min(score, 100)
    level = level_of(score)
    return score, level, reasons or ["No warning signs"], STEPS[lang][level], drivers


def meter(score, level):
    color = {"HIGH": "#e53935", "MEDIUM": "#fb8c00", "LOW": "#43a047"}[level]
    return (
        f"<div style='padding:14px;border-radius:12px;border:2px solid {color}'>"
        f"<div style='font-size:22px;font-weight:700;color:{color}'>{level} RISK: {score:.0f}/100</div>"
        f"<div style='height:14px;background:rgba(128,128,128,.25);border-radius:8px;margin-top:8px'>"
        f"<div style='height:14px;width:{score:.0f}%;background:{color};border-radius:8px'></div>"
        f"</div></div>"
    )


# =====================================================================
# 6. SESSION HISTORY / DASHBOARD
# =====================================================================
HISTORY = []
HIST_COLS = ["Time", "Amount", "UPI ID", "Risk level", "Score"]


def history_df():
    return pd.DataFrame(HISTORY, columns=HIST_COLS)


def stats_html():
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for row in HISTORY:
        counts[row[3]] += 1
    total = max(len(HISTORY), 1)
    colors = {"HIGH": "#e53935", "MEDIUM": "#fb8c00", "LOW": "#43a047"}
    html = f"<b>Checks this session: {len(HISTORY)}</b>"
    for k, v in counts.items():
        pct = v / total * 100
        html += (
            f"<div style='margin-top:8px'>{k}: {v}"
            f"<div style='height:12px;background:rgba(128,128,128,.25);border-radius:6px'>"
            f"<div style='height:12px;width:{pct:.0f}%;background:{colors[k]};border-radius:6px'></div>"
            f"</div></div>"
        )
    return html


def export_csv():
    if not HISTORY:
        return None
    path = "fraudshield_history.csv"
    history_df().to_csv(path, index=False)
    return path


# =====================================================================
# 7. HANDLERS
# =====================================================================
def run_check(amount, hour, avg, txns, new_device, new_receiver, upi, sms, lang):
    txn = {
        "amount": float(amount or 0),
        "hour": int(hour),
        "avg_spend_30d": max(float(avg or 1), 1),
        "txns_last_24h": int(txns or 0),
        "new_device": int(bool(new_device)),
        "new_receiver": int(bool(new_receiver)),
    }
    score, level, reasons, steps, drivers = combine(txn, upi, sms, lang)
    HISTORY.append([datetime.now().strftime("%H:%M:%S"), txn["amount"],
                    (upi or "-").strip() or "-", level, round(score)])
    why = "### Why\n" + "\n".join("- " + r for r in reasons)
    nxt = "### Next steps\n" + "\n".join("- " + s for s in steps)
    if drivers:
        drv = "### What pushed the risk up (AI)\n" + "\n".join(
            f"- {name}: +{d * 100:.0f} points" for name, d in drivers[:4])
    else:
        drv = "### What pushed the risk up (AI)\n- Nothing unusual in the payment pattern"
    return meter(score, level), why, nxt, drv


def sms_scan(text):
    if not text or not text.strip():
        return "<i>Paste an SMS first.</i>", ""
    s = sms_scam_check(text)
    body = "### Why\n" + ("\n".join("- " + r for r in s["reasons"]) or "- No warning signs")
    if s["urls"]:
        body += "\n\n### Links found\n" + "\n".join(f"- `{u}`" for u in s["urls"])
    return meter(s["risk_score"], s["level"]), body


def upi_scan(upi_id):
    if not upi_id or not upi_id.strip():
        return "Enter a UPI ID."
    info = upi_check(upi_id)
    icon = {"VERIFIED": "✅", "REPORTED": "🚨", "SUSPICIOUS": "⚠️",
            "UNVERIFIED": "❔", "INVALID": "❌"}[info["status"]]
    return f"### {icon} {info['status']}\n{info['msg']}"


def report_scam(upi_id, kind):
    u = (upi_id or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9.\-_]{2,}@[a-z]{2,}", u):
        return "Please enter a valid UPI ID."
    if u in REPORTED:
        REPORTED[u]["count"] += 1
    else:
        REPORTED[u] = {"count": 1, "type": kind}
    save_db()
    return f"Thank you. {u} now has {REPORTED[u]['count']} report(s)."


def otp_generate():
    code = str(100000 + secrets.randbelow(900000))
    salt = secrets.token_hex(8)
    state = {"salt": salt, "hash": hashlib.sha256((salt + code).encode()).hexdigest(),
             "created": time.time(), "tries": 0}
    return state, f"Demo OTP: **{code}**  (valid 60 seconds, 3 tries). Real OTPs must NEVER be shared."


def otp_verify(state, entered):
    if not state:
        return "Generate an OTP first."
    if time.time() - state["created"] > 60:
        return "❌ OTP expired"
    if state["tries"] >= 3:
        return "🔒 Too many attempts. Generate a new OTP."
    state["tries"] += 1
    h = hashlib.sha256((state["salt"] + (entered or "").strip()).encode()).hexdigest()
    if hmac.compare_digest(h, state["hash"]):
        return "✅ Verified"
    return f"❌ Wrong OTP ({3 - state['tries']} tries left)"


def model_info():
    m = "\n".join(f"- **{k}:** {v:.2f}" for k, v in METRICS.items())
    f = "\n".join(f"- {PRETTY.get(k, 'Amount vs average')}: {v * 100:.0f}%" for k, v in IMPORTANCES)
    return (f"### Model performance (synthetic test data)\n{m}\n\n"
            f"### Feature importance\n{f}\n\n"
            "_Trained on simulated data with 2% label noise. A real system would use real, "
            "labelled transactions._")


RISKY = [2500, 2, 60, 4, True, True, "winner.prize@upi",
         "Urgent! You won a prize. Pay processing fee at http://bit.ly/x"]
SAFE = [50, 14, 80, 2, False, False, "amazon@apl", ""]

TIPS = """
### Stay safe from UPI and SMS fraud
- Banks never ask for your **OTP, PIN or CVV**, on call or SMS.
- You never need to enter a PIN to **receive** money.
- Don't click shortened or unknown links in SMS.
- Never install apps like AnyDesk/TeamViewer for a "support agent".
- Urgent threats ("account blocked in 24 hours") are a classic scam tactic.
- Report fraud quickly: call **1930** or visit **cybercrime.gov.in**.
"""

# =====================================================================
# 8. UI
# =====================================================================
CSS = """
.gradio-container {max-width: 1000px !important; margin: auto !important;}
.hero {background: linear-gradient(135deg,#4f46e5,#7c3aed 60%,#db2777);
       padding: 28px 20px; border-radius: 18px; text-align: center; margin-bottom: 12px;}
.hero h1 {margin: 0; font-size: 30px; color: #fff !important;}
.hero p {margin: 6px 0 0; color: #fff !important; opacity: .92;}
.hero .badges span {display: inline-block; background: rgba(255,255,255,.2); color: #fff;
       border-radius: 999px; padding: 4px 12px; margin: 10px 4px 0; font-size: 13px;}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 16px;
       padding: 16px !important; box-shadow: 0 4px 14px rgba(0,0,0,.08);}
button {border-radius: 12px !important;}
button.primary {font-weight: 600 !important;}
.home h2 {text-align: center; margin: 8px 0 4px; font-size: 26px;}
.home .sub {text-align: center; opacity: .8; margin-bottom: 16px;}
.features {display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px;}
.feature {border: 1px solid rgba(128,128,128,.25); border-radius: 16px; padding: 16px;
          box-shadow: 0 4px 14px rgba(0,0,0,.06);}
.feature .ico {font-size: 28px;}
.feature b {display: block; margin: 6px 0 2px;}
.feature span {font-size: 14px; opacity: .8;}
.steps {display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-top: 8px;}
.step {border-left: 4px solid #7c3aed; padding: 8px 12px; background: rgba(124,58,237,.08); border-radius: 8px;}
.safe-note {margin-top: 16px; padding: 12px 14px; border-radius: 12px; background: rgba(229,57,53,.08);
            border: 1px solid rgba(229,57,53,.35); font-size: 14px;}
"""

HERO = """
<div class='hero'>
  <h1>🛡️ FraudShield AI Pro</h1>
  <p>Check before you pay. Scan payments, UPI IDs and SMS for fraud.</p>
  <div class='badges'><span>🤖 AI risk score</span><span>💬 SMS scanner</span>
  <span>🏦 UPI checker</span><span>🌐 English · हिंदी · ಕನ್ನಡ</span></div>
</div>
"""

HOME_HTML = """
<div class='home'>
  <h2>Don't get scammed. Check first.</h2>
  <p class='sub'>One place to test a payment, a UPI ID or a suspicious SMS before you act.</p>
  <div class='features'>
    <div class='feature'><div class='ico'>🤖</div><b>AI risk score</b>
      <span>Scores a payment from 0 to 100 and explains what raised the risk.</span></div>
    <div class='feature'><div class='ico'>💬</div><b>SMS scanner</b>
      <span>Spots fake bank alerts, scam links, prize bait and pressure tactics.</span></div>
    <div class='feature'><div class='ico'>🏦</div><b>UPI checker</b>
      <span>Checks if a UPI ID is verified, suspicious or reported by others.</span></div>
    <div class='feature'><div class='ico'>🌐</div><b>3 languages</b>
      <span>Advice in English, हिंदी and ಕನ್ನಡ.</span></div>
  </div>
  <h3 style='margin-top:18px'>How it works</h3>
  <div class='steps'>
    <div class='step'><b>1. Enter details</b><br>Payment, UPI ID or SMS text.</div>
    <div class='step'><b>2. We analyse</b><br>ML model plus scam-pattern rules.</div>
    <div class='step'><b>3. You decide</b><br>Risk level, reasons and next steps.</div>
  </div>
  <div class='safe-note'>🔒 Never share your OTP, PIN or CVV with anyone.
  Report fraud: call <b>1930</b> or visit <b>cybercrime.gov.in</b>.</div>
</div>
"""

THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="purple")
    
with gr.Blocks(title="FraudShield AI Pro") as demo:
    gr.Markdown("# 🛡️ FraudShield AI Pro\nCheck a transaction, UPI ID or SMS for fraud risk. "
                "_(Demo on synthetic data)_")

    with gr.Tabs():
        # ---------- Home ----------
        with gr.Tab("🏠 Home", id="home"):
            gr.HTML(HOME_HTML)
             
        # ---------- Full check ----------
        with gr.Tab("🔍 Full Check",id="check"):
            with gr.Row():
                with gr.Column():
                    amount = gr.Number(label="Amount (₹)", value=2500)
                    hour = gr.Slider(0, 23, step=1, value=2, label="Hour of day")
                    avg = gr.Number(label="Average spend (30 days)", value=60)
                    txns = gr.Number(label="Transactions in last 24h", value=4)
                    new_device = gr.Checkbox(label="New device", value=True)
                    new_receiver = gr.Checkbox(label="First-time receiver", value=True)
                    upi = gr.Textbox(label="UPI ID (optional)", value="winner.prize@upi")
                    sms = gr.Textbox(label="SMS text (optional)", lines=3)
                    lang = gr.Dropdown(list(STEPS.keys()), value="English", label="Language")
                    with gr.Row():
                        risky_btn = gr.Button("Load risky example")
                        safe_btn = gr.Button("Load safe example")
                    check_btn = gr.Button("Check risk", variant="primary")
                with gr.Column():
                    out_meter = gr.HTML()
                    out_why = gr.Markdown()
                    out_steps = gr.Markdown()
                    out_drv = gr.Markdown()

            inputs = [amount, hour, avg, txns, new_device, new_receiver, upi, sms]
            check_btn.click(run_check, inputs + [lang], [out_meter, out_why, out_steps, out_drv])
            risky_btn.click(lambda: RISKY, None, inputs)
            safe_btn.click(lambda: SAFE, None, inputs)

        # ---------- SMS scanner ----------
        with gr.Tab("💬 SMS Scanner",id="sms"):
            sms_in = gr.Textbox(label="Paste SMS", lines=4)
            sms_btn = gr.Button("Scan SMS", variant="primary")
            sms_meter = gr.HTML()
            sms_out = gr.Markdown()
            sms_btn.click(sms_scan, sms_in, [sms_meter, sms_out])

        # ---------- UPI checker ----------
        with gr.Tab("🏦 UPI Checker",id="upi"):
            upi_in = gr.Textbox(label="UPI ID", placeholder="name@bank")
            upi_btn = gr.Button("Check UPI ID", variant="primary")
            upi_out = gr.Markdown()
            upi_btn.click(upi_scan, upi_in, upi_out)

        # ---------- Report scam ----------
        with gr.Tab("📢 Report Scam"):
            rep_upi = gr.Textbox(label="Scammer UPI ID")
            rep_kind = gr.Dropdown(["Lottery scam", "Fake KYC", "Fake refund", "Fake job",
                                    "Other"], value="Other", label="Type of scam")
            rep_btn = gr.Button("Submit report", variant="primary")
            rep_out = gr.Markdown()
            rep_btn.click(report_scam, [rep_upi, rep_kind], rep_out)

        # ---------- OTP demo ----------
        with gr.Tab("🔐 OTP Demo"):
            otp_state = gr.State(None)
            gen_btn = gr.Button("Generate demo OTP")
            otp_msg = gr.Markdown()
            otp_in = gr.Textbox(label="Enter OTP")
            ver_btn = gr.Button("Verify", variant="primary")
            otp_res = gr.Markdown()
            gen_btn.click(otp_generate, None, [otp_state, otp_msg])
            ver_btn.click(otp_verify, [otp_state, otp_in], otp_res)

        # ---------- Dashboard ----------
        with gr.Tab("📊 Dashboard"):
            dash_btn = gr.Button("Refresh", variant="primary")
            dash_stats = gr.HTML()
            dash_table = gr.Dataframe(value=history_df(), interactive=False)
            dash_btn.click(lambda: (stats_html(), history_df()), None, [dash_stats, dash_table])
            
        

        # ---------- Tips ----------
        with gr.Tab("💡 Safety Tips"):
            gr.Markdown(TIPS)

if __name__ == "__main__":
    try:
        demo.launch(theme=THEME,css=CSS,share=True)
    except TypeError:
        demo.launch(share=True)  # use demo.launch(share=True) for a public link
