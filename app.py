"""Booth demo:  streamlit run app.py            (dev: all tabs)
             streamlit run app.py -- --booth  (booth: no data-collection tab)"""
import csv
import html
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

from textprep import PLACEHOLDERS, clean, mask_phone, tokenize

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "messages.csv"
MODEL = ROOT / "model.joblib"
METRICS = ROOT / "metrics.json"
ORIGINS = ["มือถือตัวเอง", "มือถือพ่อแม่/ญาติ", "กลุ่มเพื่อน", "ตำรวจไซเบอร์/AOC 1441",
           "ธปท./ธนาคาร", "ข่าว", "ขนส่ง/ร้านค้า"]
USER_TEST = ROOT / "data" / "user_test.csv"
# Visitors at the booth must not be able to write into the training data.
BOOTH = "--booth" in sys.argv or not DATA.exists()  # the cloud deploy ships without data/
# User-testing period only: ask testers to judge each result, for the report's evaluation section.
FEEDBACK = "--feedback" in sys.argv

PLACEHOLDER_TH = {"xurl": "มีลิงก์", "xlineid": "ชวนแอดไลน์", "xmoney": "พูดถึงจำนวนเงิน",
                  "xphone": "มีเบอร์โทร", "xnum": "มีตัวเลข"}

# Booth shortcuts: real SMS from the dataset, one genuine and one scam per topic, so visitors
# guess which is fake before the model answers. They are in the training data, so these scores
# are a demo, not evidence of accuracy (that is the per-origin test in train.py).
# Order matters: button i goes to column i % 4, which stacks each topic's pair in one column.
EXAMPLES = {
    "พัสดุ ก": "พัสดุหมายเลข 6226265185488 จัดส่งสำเร็จแล้ว",
    "ธนาคาร ก": "【KTB】คุณได้รับสิทธิ์ยื่นกู้ 200,000 บาท คลิ๊ก cutt.ly/kZaupwZ",
    "โปรโมชัน ก": "พิเศษเฉพาะคุณ! รับฟรีคูปองส่วนลดรวมสูงสุด 50บ. เมื่อช้อปครบทุก 100บ.* ที่บิ๊กซีมินิ ถึง 31 ต.ค.69 คลิก bit.ly/40SAMbS",
    "หน่วยงานรัฐ ก": "การคืนเงินประกันการใช้ไฟฟ้า การไฟฟ้าส่วนภูมิภาคPEA ยืนยัน การลงทะเบียน เงื่อนไขการขอคืนเงินประกันฯ สอบถามเพิ่มเติมที่ ... bit.ly/3Bj872Z",
    "พัสดุ ข": "ขนส่งไม่สามารถจัดส่งพัสดุของคุณได้ เนื่องจากติดต่อผู้รับไม่ได้ ติดต่อเจ้าหน้าที่ยืนยันจัดส่งอีกครั้ง: www.for-sh.cc",
    "ธนาคาร ข": "เงินโอนเข้าบ/ชX1234 ผ่านระบบ 5,000.00บ ใช้ได้ 15,000.00บ@07:15",
    "โปรโมชัน ข": "ยินดีด้วย คุณได้รับซองแดงกับ 5977 บาท คลิก cutt.ly/uedEGWlg",
    "หน่วยงานรัฐ ข": "ลงทะเบียนคนละครึ่งพลัสสำเร็จ คุณสามารถใช้สิทธิผ่านแอปฯ เป๋าตังได้ตั้งแต่วันที่ 29 ต.ค. 68 เป็นต้นไป",
}

st.set_page_config(page_title="เช็ค SMS มิจฉาชีพ", page_icon="🛡️", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL) if MODEL.exists() else None


def use_example(text):
    # Runs before the rerun, so the text area picks up the new value and the check fires at once.
    st.session_state["msg"] = text
    st.session_state["auto_check"] = True


def save_feedback(last, truth, helpful, consent):
    # The message text is kept only with explicit consent; otherwise just the judgement.
    new = not USER_TEST.exists()
    with USER_TEST.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "prob_scam", "is_example", "truth", "helpful", "text"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), f"{last['p']:.3f}",
                    last["text"] in EXAMPLES.values(), truth, helpful,
                    mask_phone(last["text"]).replace("\n", " ") if consent else ""])


def contributions(pipe, text):
    """Per-feature push toward 'scam' for one message (works for LR and NB)."""
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    x = vec.transform([text])
    if hasattr(clf, "coef_"):
        w = clf.coef_[0]
    elif hasattr(clf, "feature_log_prob_"):
        w = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    else:
        return {}
    names = vec.get_feature_names_out()
    idx = x.nonzero()[1]
    return {names[i]: float(x[0, i] * w[i]) for i in idx}


def highlight(text, contrib):
    toks = tokenize(text)
    top = max((abs(v) for v in contrib.values()), default=1.0) or 1.0
    out = []
    for t in toks:
        c = contrib.get(t, 0.0)
        label = PLACEHOLDER_TH.get(t, t)
        if c > 0.02 * top:
            a = 0.18 + 0.62 * min(1, c / top)
            out.append(f'<span style="background:rgba(239,68,68,{a:.2f});padding:1px 3px;border-radius:4px">{html.escape(label)}</span>')
        elif c < -0.02 * top:
            a = 0.12 + 0.45 * min(1, -c / top)
            out.append(f'<span style="background:rgba(34,197,94,{a:.2f});padding:1px 3px;border-radius:4px">{html.escape(label)}</span>')
        else:
            out.append(html.escape(label))
    return " ".join(out)


pipe = load_model()
if BOOTH:
    tab_check, tab_about = st.tabs(["🛡️ ตรวจข้อความ", "📊 เกี่ยวกับโมเดล"])
    tab_collect = None
else:
    tab_check, tab_collect, tab_about = st.tabs(["🛡️ ตรวจข้อความ", "➕ เก็บข้อมูล", "📊 เกี่ยวกับโมเดล"])

with tab_check:
    st.title("🛡️ ข้อความนี้มิจฉาชีพหรือเปล่า?")
    st.caption("วาง SMS ที่ได้รับ แล้วกดตรวจ (ออกแบบมาสำหรับ SMS ไม่ใช่แชทส่วนตัว)")
    if pipe is None:
        st.error("ยังไม่มี model.joblib - รัน `python train.py` ก่อน")
    else:
        st.caption("หรือกดลองตัวอย่าง SMS จริง - แต่ละหัวข้อมีของจริง 1 ของหลอก 1 ทายก่อนว่าอันไหนหลอก")
        cols = st.columns(4)
        for i, (name, example) in enumerate(EXAMPLES.items()):
            cols[i % 4].button(name, key=f"example_{i}", on_click=use_example, args=(example,),
                               use_container_width=True)

        txt = st.text_area("ข้อความ", key="msg", height=140,
                           placeholder="เช่น พัสดุของท่านถูกกักไว้ที่ศุลกากร กรุณาชำระค่าธรรมเนียม...")
        clicked = st.button("ตรวจ", type="primary", use_container_width=True)
        auto = st.session_state.pop("auto_check", False)
        checked = (clicked or auto) and txt.strip()
        if checked and pipe.named_steps["tfidf"].transform([txt]).nnz == 0:
            # No known word at all: the score would just be the class prior (~50%), not a judgement.
            st.info("โมเดลไม่รู้จักคำในข้อความนี้เลย ตัดสินไม่ได้ - ลองวาง SMS ทั้งข้อความ")
        elif checked:
            p = float(pipe.predict_proba([txt])[0][1])
            st.session_state["last"] = {"text": txt, "p": p}
            if p >= 0.65:
                st.error(f"### 🚨 เสี่ยงสูง - {p:.0%}\nอย่ากดลิงก์ อย่าโอนเงิน อย่าให้ OTP · โทรเช็คกับหน่วยงานจากเบอร์ทางการเอง")
            elif p >= 0.35:
                st.warning(f"### ⚠️ น่าสงสัย - {p:.0%}\nตรวจสอบผู้ส่งก่อนทำตามที่ข้อความบอก")
            else:
                st.success(f"### ✅ น่าจะปลอดภัย - {p:.0%}")
            st.progress(p)

            c = contributions(pipe, txt)
            st.markdown("**คำที่ทำให้โมเดลคิดแบบนี้** - 🟥 ดันไปทางมิจฉาชีพ · 🟩 ดันไปทางปกติ")
            st.markdown(f'<div style="line-height:2.1;font-size:1.05rem">{highlight(txt, c)}</div>', unsafe_allow_html=True)

            pos = sorted(((k, v) for k, v in c.items() if v > 0), key=lambda kv: -kv[1])[:5]
            if pos:
                st.markdown("**เหตุผลหลัก**")
                for k, v in pos:
                    st.markdown(f"- `{PLACEHOLDER_TH.get(k, k)}`  (+{v:.2f})")
            st.caption("โมเดลเป็นตัวช่วยตัดสินใจ ไม่ใช่คำตัดสินสุดท้าย - ถ้าไม่แน่ใจ โทรถามหน่วยงานจากเบอร์ทางการเสมอ")

            # The brief asks the demo to show the process, not only the answer.
            with st.expander("ดูขั้นตอนที่โมเดลทำ"):
                st.markdown("**1. แทนลิงก์ เบอร์ เงิน ด้วยคำกลาง**")
                st.code(clean(txt), language=None)
                st.markdown("**2. ตัดคำภาษาไทย** (pythainlp newmm)")
                st.code(" | ".join(tokenize(txt)), language=None)
                x = pipe.named_steps["tfidf"].transform([txt])
                st.markdown(f"**3. แปลงเป็นตัวเลขด้วย TF-IDF** - ใช้ได้ {x.nnz} คำ/คู่คำ "
                            f"จากคำศัพท์ทั้งหมด {x.shape[1]:,} ที่โมเดลรู้จัก")
                st.markdown(f"**4. {type(pipe.named_steps['clf']).__name__}** - ความน่าจะเป็นว่าเป็นมิจฉาชีพ "
                            f"= {p:.0%} · เกณฑ์: ต่ำกว่า 35% ปลอดภัย · 35-65% น่าสงสัย · 65% ขึ้นไป เสี่ยงสูง")

        if FEEDBACK and "last" in st.session_state:
            with st.form("feedback", clear_on_submit=True):
                st.markdown("**ช่วยตอบ 2 ข้อ** (ใช้วัดผลในรายงาน)")
                truth = st.radio("ข้อความนี้จริงๆ เป็นอะไร", ["มิจฉาชีพ", "ปกติ", "ไม่แน่ใจ"], horizontal=True)
                helpful = st.radio("คำที่ไฮไลต์ช่วยให้เข้าใจไหม", ["ช่วย", "ไม่ช่วย"], horizontal=True)
                consent = st.checkbox("ยินยอมให้เก็บข้อความนี้ไปพัฒนาโมเดล (ปิดเบอร์โทรให้อัตโนมัติ)")
                st.caption("ถ้าไม่ติ๊ก จะเก็บแค่คำตอบ 2 ข้อ ไม่เก็บตัวข้อความ")
                if st.form_submit_button("ส่ง"):
                    save_feedback(st.session_state.pop("last"), truth, helpful, consent)
                    st.success("ขอบคุณ - ลองข้อความถัดไปได้เลย")

if tab_collect is not None:
    with tab_collect:
        st.subheader("เพิ่มข้อความลง dataset")
        st.caption("ใช้ช่วงพัฒนาเท่านั้น · เบอร์โทรจะถูกปิดบังอัตโนมัติ · อย่าใส่ชื่อ เลขบัญชี หรือเลขบัตรประชาชน")
        # Outside the form so it survives clear_on_submit while entering a batch from one source.
        origin = st.selectbox("มาจากไหน", ORIGINS, accept_new_options=True,
                              help="ใช้แยก test ตามแหล่ง - พิมพ์ชื่อแหล่งใหม่ได้")
        with st.form("add", clear_on_submit=True):
            t = st.text_area("ข้อความ", height=110)
            lab = st.radio("ประเภท", ["scam", "normal"], horizontal=True,
                           format_func=lambda s: "มิจฉาชีพ" if s == "scam" else "ปกติ")
            if st.form_submit_button("บันทึก") and t.strip():
                DATA.parent.mkdir(exist_ok=True)
                new = not DATA.exists()
                with DATA.open("a", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(["text", "label", "source", "origin"])
                    w.writerow([mask_phone(t.strip()).replace("\n", " "), lab, "real", origin])
                st.success("บันทึกแล้ว - รัน `python train.py` ใหม่เมื่อเก็บได้พอ")
        if DATA.exists():
            import pandas as pd
            d = pd.read_csv(DATA)
            st.write(d.groupby(["origin", "label"]).size().unstack(fill_value=0))

with tab_about:
    if METRICS.exists():
        m = json.loads(METRICS.read_text(encoding="utf-8"))
        st.write(f"**โมเดล:** {m['model']} · ข้อมูล {m['n_total']} ข้อความ "
                 f"(จริง {sum(m['n_real'].values())} · ตัวอย่าง/seed {m['n_seed']})")
        t = m.get("test_on_real")
        if t:
            a, b, c = st.columns(3)
            a.metric("Recall (จับ scam ได้)", f"{t['recall_scam']:.0%}")
            b.metric("Precision", f"{t['precision_scam']:.0%}")
            c.metric("F1", f"{t['f1_scam']:.2f}")
            st.caption(f"วัดบนข้อความจริงที่โมเดลไม่เคยเห็น {t['n_test']} ข้อความ")
        else:
            st.warning("ยังไม่มีผลทดสอบบนข้อมูลจริง - เก็บข้อมูลเพิ่มแล้ว train ใหม่")
    else:
        st.info("รัน `python train.py` ก่อน")
