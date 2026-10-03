"""Booth demo:  streamlit run app.py            (dev: all tabs)
             streamlit run app.py -- --booth  (booth: no data-collection tab)
Colours come from .streamlit/config.toml, to match the brochure."""
import csv
import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from textprep import URL_RE, clean, mask_phone, tokenize

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "messages.csv"
MODEL = ROOT / "model.joblib"
METRICS = ROOT / "metrics.json"
SUMMARY = ROOT / "results" / "summary.json"
ORIGINS = ["มือถือตัวเอง", "มือถือพ่อแม่/ญาติ", "กลุ่มเพื่อน", "ตำรวจไซเบอร์/AOC 1441",
           "ธปท./ธนาคาร", "ข่าว", "ขนส่ง/ร้านค้า"]
USER_TEST = ROOT / "data" / "user_test.csv"
# Visitors at the booth must not be able to write into the training data.
BOOTH = "--booth" in sys.argv or not DATA.exists()  # the cloud deploy ships without data/
# User-testing period only: ask testers to judge each result, for the report's evaluation section.
FEEDBACK = "--feedback" in sys.argv

PLACEHOLDER_TH = {"xurl": "มีลิงก์", "xlineid": "ชวนแอดไลน์", "xmoney": "พูดถึงจำนวนเงิน",
                  "xphone": "มีเบอร์โทร", "xnum": "มีตัวเลข"}
# Shorteners hide where a link really goes, which is worth telling the visitor.
SHORTENERS = {"bit.ly", "cutt.ly", "tinyurl.com", "s.id", "shorturl.at", "rb.gy", "t.ly", "is.gd", "goo.gl", "t.co"}
GOAL_RECALL, GOAL_FALSE_ALARM = 0.90, 0.15
# Below this many non-space characters there is too little text to judge ("ไปไหนมา" scored 70%).
# The shortest real scam in the data is 16 characters; only 2 of 2,927 real SMS are shorter than 15.
MIN_CHARS = 15
ORIGIN_TH = {"มือถือตัวเอง": "SMS ผู้จัดทำ", "ชุดข้อมูล ScamGuard": "ชุด ScamGuard", "ชุดข้อมูล ssivakorn": "ชุด ssivakorn"}

# Booth shortcuts: real SMS from the dataset, one genuine and one scam per topic, so visitors
# guess which is fake before the model answers. They are in the training data, so these scores
# are a demo, not evidence of accuracy (that is the cross-source test in evaluate.py).
# Kept in pairs, so the buttons wrap as 4 per row on a laptop and 2 per row on a phone
# with each topic's ก and ข side by side either way.
EXAMPLES = {
    "พัสดุ ก": "พัสดุหมายเลข 6226265185488 จัดส่งสำเร็จแล้ว",
    "พัสดุ ข": "ขนส่งไม่สามารถจัดส่งพัสดุของคุณได้ เนื่องจากติดต่อผู้รับไม่ได้ ติดต่อเจ้าหน้าที่ยืนยันจัดส่งอีกครั้ง: www.for-sh.cc",
    "ธนาคาร ก": "【KTB】คุณได้รับสิทธิ์ยื่นกู้ 200,000 บาท คลิ๊ก cutt.ly/kZaupwZ",
    "ธนาคาร ข": "เงินโอนเข้าบ/ชX1234 ผ่านระบบ 5,000.00บ ใช้ได้ 15,000.00บ@07:15",
    "โปรโมชัน ก": "พิเศษเฉพาะคุณ! รับฟรีคูปองส่วนลดรวมสูงสุด 50บ. เมื่อช้อปครบทุก 100บ.* ที่บิ๊กซีมินิ ถึง 31 ต.ค.69 คลิก bit.ly/40SAMbS",
    "โปรโมชัน ข": "ยินดีด้วย คุณได้รับซองแดงกับ 5977 บาท คลิก cutt.ly/uedEGWlg",
    "หน่วยงานรัฐ ก": "การคืนเงินประกันการใช้ไฟฟ้า การไฟฟ้าส่วนภูมิภาคPEA ยืนยัน การลงทะเบียน เงื่อนไขการขอคืนเงินประกันฯ สอบถามเพิ่มเติมที่ ... bit.ly/3Bj872Z",
    "หน่วยงานรัฐ ข": "ลงทะเบียนคนละครึ่งพลัสสำเร็จ คุณสามารถใช้สิทธิผ่านแอปฯ เป๋าตังได้ตั้งแต่วันที่ 29 ต.ค. 68 เป็นต้นไป",
}

st.set_page_config(page_title="ScamSMS เช็ค SMS มิจฉาชีพ", page_icon="🛡️", layout="centered")


@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")  # the default shows "Running load_model()." to visitors
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


def feature_label(feature):
    # Features can be word pairs ("คลิ๊ก xurl"), so translate each word. Visitors paste anything,
    # so drop characters that markdown would read as syntax.
    words = " ".join(PLACEHOLDER_TH.get(w, w) for w in feature.split())
    return re.sub(r"[\[\]\\*_$`~]", "", words)


def link_hosts(text):
    """Domains of the links in a message, in order of appearance."""
    hosts = []
    for m in URL_RE.finditer(text):
        h = re.sub(r"^(https?://)?(www\.)?", "", m.group(0).lower()).split("/")[0]
        h = re.sub(r"[^\w.-]", "", h)  # shown as markdown, so keep only domain characters
        if h and h not in hosts:
            hosts.append(h)
    return hosts


def suspicious_advice(text):
    # Testers were unsure what "suspicious" meant for genuine shop SMS with a link: say why, and what to do.
    hosts = link_hosts(text)
    if not hosts:
        return "ตรวจสอบผู้ส่งก่อนทำตามที่ข้อความบอก"
    where = ", ".join(f"**{h}**" for h in hosts)
    short = " (ลิงก์ย่อ ดูไม่ออกว่าไปเว็บไหน)" if any(h in SHORTENERS for h in hosts) else ""
    return (f"มีลิงก์ไปที่ {where}{short}  \n"
            "ของจริงกับของหลอกส่งลิงก์แบบนี้ได้ทั้งคู่ โมเดลดูแค่ตัวข้อความจึงแยกไม่ออก "
            "ไม่ต้องกดลิงก์ ให้เข้าแอปหรือเว็บทางการเองแทน")


def nowrap(text):
    # Thai has no spaces, so a narrow phone screen breaks lines mid-word.
    # Keep each space-separated word whole and let lines break only at the spaces.
    return " ".join(f'<span style="white-space:nowrap">{word}</span>' for word in text.split(" "))


def highlight(text, contrib):
    toks = tokenize(text, lower=False)
    top = max((abs(v) for v in contrib.values()), default=1.0) or 1.0
    out = []
    for t in toks:
        c = contrib.get(t.lower(), 0.0)
        label = html.escape(PLACEHOLDER_TH.get(t, t))
        style = "padding:1px 3px;border-radius:4px;white-space:nowrap"
        if c > 0.02 * top:
            a = 0.18 + 0.5 * min(1, c / top)
            out.append(f'<span style="background:rgba(214,59,59,{a:.2f});{style}">{label}</span>')
        elif c < -0.02 * top:
            a = 0.12 + 0.4 * min(1, -c / top)
            out.append(f'<span style="background:rgba(26,127,75,{a:.2f});{style}">{label}</span>')
        else:
            out.append(label)
    return " ".join(out)


st.title(":blue[ScamSMS]")
st.markdown("**เช็ค SMS มิจฉาชีพภาษาไทย ด้วย Machine&nbsp;Learning**")
st.caption("Mini Project 240-318 AI-ML")

pipe = load_model()
if BOOTH:
    tab_check, tab_about = st.tabs(["ตรวจข้อความ", "เกี่ยวกับโมเดล"])
    tab_collect = None
else:
    tab_check, tab_collect, tab_about = st.tabs(["ตรวจข้อความ", "เก็บข้อมูล", "เกี่ยวกับโมเดล"])

with tab_check:
    if pipe is None:
        st.error("ยังไม่มี model.joblib รัน `python train.py` ก่อน")
    else:
        st.markdown("**ลองตัวอย่าง SMS จริง**  \nแต่ละหัวข้อมีของจริง 1 ของหลอก 1 ลองทายดูก่อน")
        # A wrapping row instead of st.columns, which would stack all 8 buttons on a phone.
        # 160px fits 4 per row in the 704px centered layout and 2 per row on a 390px phone.
        with st.container(horizontal=True, horizontal_alignment="distribute"):
            for i, (name, example) in enumerate(EXAMPLES.items()):
                st.button(name, key=f"example_{i}", on_click=use_example, args=(example,), width=160)

        txt = st.text_area("วาง SMS ที่ได้รับ", key="msg", height=140,
                           placeholder="เช่น พัสดุของท่านถูกกักไว้ที่ศุลกากร กรุณาชำระค่าธรรมเนียม...")
        clicked = st.button("ตรวจ", type="primary", width="stretch")
        auto = st.session_state.pop("auto_check", False)
        checked = (clicked or auto) and txt.strip()
        if checked and len(re.sub(r"\s", "", txt)) < MIN_CHARS:
            st.info("ข้อความสั้นเกินไป จึงตัดสินไม่ได้ ลองวาง SMS ทั้งข้อความ")
        elif checked and pipe.named_steps["tfidf"].transform([txt]).nnz == 0:
            # No known word at all: the score would just be the class prior (~50%), not a judgement.
            st.info("โมเดลไม่รู้จักคำในข้อความนี้เลย จึงตัดสินไม่ได้ ลองวาง SMS ทั้งข้อความ")
        elif checked:
            p = float(pipe.predict_proba([txt])[0][1])
            st.session_state["last"] = {"text": txt, "p": p}
            # Spelled out, because testers read "ปลอดภัย - 5%" as minus five percent, or as "5% safe".
            chance = f"**โอกาสเป็นมิจฉาชีพ {p:.0%}**"
            if p >= 0.65:
                st.error(f"### เสี่ยงสูง\n{chance}  \n"
                         "อย่ากดลิงก์ อย่าโอนเงิน อย่าให้ OTP ถ้าไม่แน่ใจให้โทรถามหน่วยงานจากเบอร์ทางการเอง")
            elif p >= 0.35:
                st.warning(f"### น่าสงสัย\n{chance}  \n{suspicious_advice(txt)}")
            else:
                st.success(f"### น่าจะปลอดภัย\n{chance}")

            c = contributions(pipe, txt)
            with st.container(border=True):
                st.markdown("**คำที่ทำให้โมเดลคิดแบบนี้**  \n"
                            ":red-background[ดันไปทางมิจฉาชีพ] &nbsp; :green-background[ดันไปทางปกติ]")
                st.markdown(f'<div style="line-height:2.1;font-size:1.05rem">{highlight(txt, c)}</div>', unsafe_allow_html=True)
            st.caption("โมเดลเป็นตัวช่วยตัดสินใจ ไม่ใช่คำตัดสินสุดท้าย")

            # The brief asks the demo to show the process, not only the answer.
            with st.expander("ดูขั้นตอนที่โมเดลทำ"):
                st.markdown("**1. แทนลิงก์ เบอร์ เงิน ด้วยคำกลาง**")
                st.code(clean(txt), language=None)
                st.markdown("**2. ตัดคำภาษาไทย** (pythainlp newmm)")
                st.code(" | ".join(tokenize(txt)), language=None)
                x = pipe.named_steps["tfidf"].transform([txt])
                st.markdown(f"**3. แปลงเป็นตัวเลขด้วย TF-IDF**  \n"
                            f"เจอคำ/คู่คำที่โมเดลรู้จัก {x.nnz} ตัว จากคำศัพท์ทั้งหมด {x.shape[1]:,} ตัว")
                model_name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", type(pipe.named_steps["clf"]).__name__)
                st.markdown(f"**4. {model_name} คิดโอกาสเป็นมิจฉาชีพ {p:.0%}**  \n"
                            "เกณฑ์: ต่ำกว่า 35% น่าจะปลอดภัย, 35% ถึง 65% น่าสงสัย, 65% ขึ้นไป เสี่ยงสูง")
                pos = sorted(((k, v) for k, v in c.items() if v > 0), key=lambda kv: -kv[1])[:5]
                if pos:
                    badges = " ".join(f":red-badge[{feature_label(k)} +{v:.2f}]" for k, v in pos)
                    st.markdown(f"**5. คำที่ดันไปทางมิจฉาชีพมากที่สุด** (ค่าน้ำหนักจากโมเดล)  \n{badges}")

        if FEEDBACK and "last" in st.session_state:
            with st.form("feedback", clear_on_submit=True):
                st.markdown("**ช่วยตอบ 2 ข้อ** (ใช้วัดผลในรายงาน)")
                truth = st.radio("ข้อความนี้จริงๆ เป็นอะไร", ["มิจฉาชีพ", "ปกติ", "ไม่แน่ใจ"], horizontal=True)
                helpful = st.radio("คำที่ไฮไลต์ช่วยให้เข้าใจไหม", ["ช่วย", "ไม่ช่วย"], horizontal=True)
                consent = st.checkbox("ยินยอมให้เก็บข้อความนี้ไปพัฒนาโมเดล (ปิดเบอร์โทรให้อัตโนมัติ)")
                st.caption("ถ้าไม่ติ๊ก จะเก็บแค่คำตอบ 2 ข้อ ไม่เก็บตัวข้อความ")
                if st.form_submit_button("ส่ง"):
                    save_feedback(st.session_state.pop("last"), truth, helpful, consent)
                    st.success("ขอบคุณ ลองข้อความถัดไปได้เลย")

if tab_collect is not None:
    with tab_collect:
        st.subheader("เพิ่มข้อความลง dataset")
        st.caption("ใช้ช่วงพัฒนาเท่านั้น · เบอร์โทรจะถูกปิดบังอัตโนมัติ · อย่าใส่ชื่อ เลขบัญชี หรือเลขบัตรประชาชน")
        # Outside the form so it survives clear_on_submit while entering a batch from one source.
        origin = st.selectbox("มาจากไหน", ORIGINS, accept_new_options=True,
                              help="ใช้แยก test ตามแหล่ง พิมพ์ชื่อแหล่งใหม่ได้")
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
                st.success("บันทึกแล้ว รัน `python train.py` ใหม่เมื่อเก็บได้พอ")
        if DATA.exists():
            d = pd.read_csv(DATA)
            st.write(d.groupby(["origin", "label"]).size().unstack(fill_value=0))

with tab_about:
    if SUMMARY.exists():
        s = json.loads(SUMMARY.read_text(encoding="utf-8"))
        lr = s["random_split"]["Logistic Regression"]
        real = s["data"]["real"]
        st.write(f"**โมเดล:** TF-IDF (คำเดี่ยว + คู่คำ) + Logistic Regression · "
                 f"ข้อความจริง หลอก {real.get('scam', 0):,} · ปกติ {real.get('normal', 0):,}  \n"
                 "**ขอบเขต:** SMS ภาษาไทย ไม่รวมแชทส่วนตัวและสายโทร")
        st.markdown("**ค่าเฉลี่ยจากสุ่มแบ่ง 10 รอบ** (ข้อความจริง 25% เป็นชุดทดสอบ)")
        a, b, c = st.columns(3)
        a.metric("จับของหลอกได้ (recall)", f"{lr['recall']['mean']:.0%}", border=True)
        b.metric("เตือนแล้วถูก (precision)", f"{lr['precision']['mean']:.0%}", border=True)
        c.metric("ปกติแต่โดนเตือน", f"{lr['false_alarm_rate']['mean']:.0%}", border=True)

        st.markdown(f"**เทียบกับเป้าหมาย** (จับได้ ≥ {GOAL_RECALL:.0%} · เตือนผิด ≤ {GOAL_FALSE_ALARM:.0%})")
        rows = [(nowrap("สุ่มแบ่ง 10 รอบ (ทุกแหล่ง)"), lr["recall"]["mean"], lr["false_alarm_rate"]["mean"])]
        for origin, name in ORIGIN_TH.items():
            r = s["by_origin"][origin]["Logistic Regression"]
            rows.append((nowrap(f"ข้ามแหล่ง: {name}"), r["recall"], r["false_alarm_rate"]))
        table = f"| {nowrap('ชุดทดสอบ')} | {nowrap('จับได้')} | {nowrap('เตือนผิด')} | ผล |\n|---|--:|--:|---|\n"
        for name, rec, fa in rows:
            passed = rec >= GOAL_RECALL and fa <= GOAL_FALSE_ALARM
            result = ":green[**ผ่าน**]" if passed else nowrap(":orange[**ยังไม่ถึง**]")
            table += f"| {name} | {rec:.0%} | {fa:.0%} | {result} |\n"
        st.markdown(table, unsafe_allow_html=True)
        st.caption("ข้ามแหล่ง = train โดยไม่ใช้ข้อมูลจากแหล่งนั้น แล้วทดสอบกับแหล่งนั้น")
    elif METRICS.exists():
        m = json.loads(METRICS.read_text(encoding="utf-8"))
        st.write(f"**โมเดล:** {m['model']} · ข้อมูล {m['n_total']} ข้อความ")
        st.caption("รัน `python evaluate.py` เพื่อดูผลประเมินแบบเต็ม")
    else:
        st.info("รัน `python train.py` ก่อน")
