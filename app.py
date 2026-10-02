"""Booth demo:  streamlit run app.py            (dev: all tabs)
             streamlit run app.py -- --booth  (booth: no data-collection tab)"""
import csv
import html
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import streamlit as st

from textprep import clean, mask_phone, tokenize

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

# (lower bound, css class, label, advice) checked top to bottom
LEVELS = [
    (0.65, "high", "เสี่ยงสูง", "อย่ากดลิงก์ อย่าโอนเงิน อย่าให้ OTP · โทรเช็คกับหน่วยงานจากเบอร์ทางการเอง"),
    (0.35, "mid", "น่าสงสัย", "ตรวจสอบผู้ส่งก่อนทำตามที่ข้อความบอก"),
    (0.0, "low", "น่าจะปลอดภัย", "ไม่พบสัญญาณหลอกลวงที่ชัดเจน"),
]
GOAL_RECALL, GOAL_FALSE_ALARM = 0.90, 0.15
ORIGIN_TH = {"มือถือตัวเอง": "SMS ผู้จัดทำ", "ชุดข้อมูล ScamGuard": "ชุด ScamGuard", "ชุดข้อมูล ssivakorn": "ชุด ssivakorn"}

# Booth shortcuts: real SMS from the dataset, one genuine and one scam per topic, so visitors
# guess which is fake before the model answers. They are in the training data, so these scores
# are a demo, not evidence of accuracy (that is the cross-source test in evaluate.py).
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

# Same visual language as the brochure: navy header band, framed boxes with a tab label,
# faint SMS-bubble background. Colours and font come from .streamlit/config.toml.
BUBBLES = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='96' height='96' viewBox='0 0 96 96'%3E"
           "%3Cg fill='none' stroke='%2317345f' stroke-opacity='.06' stroke-width='1.5'%3E"
           "%3Cpath d='M14 12h28a6 6 0 0 1 6 6v11a6 6 0 0 1-6 6H26l-8 7v-7h-4a6 6 0 0 1-6-6V18a6 6 0 0 1 6-6z'/%3E"
           "%3Cpath d='M58 56h24a5 5 0 0 1 5 5v9a5 5 0 0 1-5 5h-3v6l-7-6H58a5 5 0 0 1-5-5v-9a5 5 0 0 1 5-5z'/%3E"
           "%3C/g%3E%3C/svg%3E")
STYLE = f"""<style>
[data-testid="stAppViewContainer"] {{ background-image: url("{BUBBLES}"); background-size: 96px; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"] {{ padding-top: 4rem; }}
.nw {{ white-space: nowrap; }}
.hero {{ background: #17345f; color: #fff; border-radius: 14px; padding: 18px 22px; margin-bottom: 4px; }}
.hero .kicker {{ font-size: .78rem; letter-spacing: .06em; color: #b9cdec; font-weight: 600; margin: 0; }}
.hero h1 {{ color: #fff; font-size: 2.3rem; font-weight: 800; margin: 2px 0 0; padding: 0; line-height: 1.1; }}
.hero p.sub {{ color: #e8eefa; margin: 4px 0 0; font-weight: 600; }}
.box {{ position: relative; background: #fff; border: 1.4px solid #c9d6ea; border-radius: 12px;
        padding: 20px 16px 12px; margin: 16px 0 8px; }}
.box > .tab {{ position: absolute; top: -11px; left: 14px; background: #17345f; color: #fff;
               font-size: .8rem; font-weight: 700; padding: 1px 10px; border-radius: 6px; }}
.box.high {{ border-color: #d63b3b; }} .box.high > .tab, .box.high .badge {{ background: #d63b3b; }}
.box.mid {{ border-color: #b7791f; }} .box.mid > .tab, .box.mid .badge {{ background: #b7791f; }}
.box.low {{ border-color: #1a7f4b; }} .box.low > .tab, .box.low .badge {{ background: #1a7f4b; }}
.verdict {{ display: flex; align-items: center; gap: 14px; }}
.badge {{ color: #fff; font-weight: 800; font-size: 1.35rem; padding: 4px 14px; border-radius: 8px; white-space: nowrap; }}
.meter {{ flex: 1; height: 10px; background: #eef3fa; border-radius: 6px; overflow: hidden; }}
.meter i {{ display: block; height: 100%; background: currentColor; border-radius: 6px; }}
.box.high .meter {{ color: #d63b3b; }} .box.mid .meter {{ color: #b7791f; }} .box.low .meter {{ color: #1a7f4b; }}
.box .advice {{ margin: 10px 0 0; color: #4a5466; }}
.box .legend {{ font-size: .85rem; color: #4a5466; margin: 0 0 6px; }}
.sq {{ display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin: 0 4px 0 10px; vertical-align: -1px; }}
.sq.red {{ background: rgba(214, 59, 59, .55); margin-left: 0; }} .sq.green {{ background: rgba(26, 127, 75, .45); }}
.hl-text {{ line-height: 2.1; font-size: 1.05rem; }}
.hl {{ padding: 1px 3px; border-radius: 4px; white-space: nowrap; }}
.reasons {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }}
.reasons .chip {{ background: #eef3fa; border: 1px solid #c9d6ea; border-radius: 6px; padding: 1px 9px; font-size: .88rem; }}
.reasons .chip b {{ color: #d63b3b; }}
.stMarkdown .label {{ font-weight: 700; color: #17345f; margin: 14px 0 0; }}
.kpis {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
.kpi {{ background: #eef3fa; border-radius: 10px; padding: 8px 12px; }}
.kpi .v {{ font-size: 1.8rem; font-weight: 800; color: #17345f; line-height: 1.1; margin: 0; }}
.kpi .l {{ font-size: .8rem; color: #4a5466; margin: 0; line-height: 1.35; }}
.box table {{ width: 100%; border-collapse: collapse; font-size: .92rem; }}
.box th, .box td {{ padding: 5px 8px; text-align: right; border-bottom: 1px solid #c9d6ea; white-space: nowrap; }}
.box th:first-child, .box td:first-child {{ text-align: left; white-space: normal; }}
.box th {{ color: #4a5466; font-weight: 600; }}
.pass {{ color: #1a7f4b; font-weight: 700; }} .miss {{ color: #b7791f; font-weight: 700; }}
.box .note {{ font-size: .8rem; color: #4a5466; margin: 6px 0 0; }}

/* Phones. Thai has no spaces, so narrow cells break mid-phrase; give text room instead of squeezing it. */
@media (max-width: 640px) {{
  /* Streamlit stacks all 4 columns into 8 full-width rows; keep two topics per row instead. */
  .st-key-examples [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; gap: 1rem; }}
  .st-key-examples [data-testid="stColumn"] {{ flex: 1 1 calc(50% - .5rem) !important;
                                               min-width: calc(50% - .5rem) !important; }}
  .kpis {{ grid-template-columns: 1fr; gap: 6px; }}
  .kpi {{ display: flex; align-items: center; gap: 14px; }}
  .kpi .v {{ min-width: 72px; }}
  .box table {{ font-size: .85rem; }}
  .box th, .box td {{ padding: 5px 4px; }}
}}
</style>"""

st.set_page_config(page_title="ScamSMS เช็ค SMS มิจฉาชีพ", page_icon="🛡️", layout="centered")
st.markdown(STYLE, unsafe_allow_html=True)


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
        label = html.escape(PLACEHOLDER_TH.get(t, t))
        if c > 0.02 * top:
            a = 0.18 + 0.5 * min(1, c / top)
            out.append(f'<span class="hl" style="background:rgba(214,59,59,{a:.2f})">{label}</span>')
        elif c < -0.02 * top:
            a = 0.12 + 0.4 * min(1, -c / top)
            out.append(f'<span class="hl" style="background:rgba(26,127,75,{a:.2f})">{label}</span>')
        else:
            out.append(label)
    return " ".join(out)


def box(tab, body, kind=""):
    """A framed box with a tab label, like the brochure. Kept on one line so markdown leaves it alone."""
    return f'<div class="box {kind}"><span class="tab">{tab}</span>{body}</div>'


def verdict_box(p):
    for cut, kind, label, advice in LEVELS:
        if p >= cut:
            break
    body = (f'<div class="verdict"><span class="badge">{label} {p:.0%}</span>'
            f'<div class="meter"><i style="width:{p * 100:.0f}%"></i></div></div>'
            f'<p class="advice">{advice}</p>')
    return box("ผลการตรวจ", body, kind)


def reasons_box(text, contrib):
    legend = ('<p class="legend"><i class="sq red"></i>ดันไปทางมิจฉาชีพ'
              '<i class="sq green"></i>ดันไปทางปกติ</p>')
    body = legend + f'<div class="hl-text">{highlight(text, contrib)}</div>'
    top = sorted(((k, v) for k, v in contrib.items() if v > 0), key=lambda kv: -kv[1])[:5]
    if top:
        # features can be word pairs ("คลิ๊ก xurl"), so translate each word, not the whole key
        chips = "".join(f'<span class="chip">{html.escape(" ".join(PLACEHOLDER_TH.get(w, w) for w in k.split()))}'
                        f' <b>+{v:.2f}</b></span>' for k, v in top)
        body += f'<p class="label" style="margin-top:12px">เหตุผลหลัก</p><div class="reasons">{chips}</div>'
    return box("คำที่ทำให้โมเดลคิดแบบนี้", body)


def about_html(s):
    lr = s["random_split"]["Logistic Regression"]
    kpis = "".join(
        f'<div class="kpi"><p class="v">{lr[key]["mean"]:.0%}</p>'
        f'<p class="l"><span class="nw">{th}</span><br><span class="nw">{en} ± {lr[key]["std"] * 100:.1f}</span></p></div>'
        for key, th, en in [("recall", "จับของหลอกได้", "recall"), ("precision", "เตือนแล้วถูก", "precision"),
                            ("false_alarm_rate", "ปกติแต่โดนเตือน", "false alarm")])
    # nowrap pieces so a narrow first column breaks between phrases, not inside a Thai word
    rows = [('<span class="nw">สุ่มแบ่ง 10 รอบ</span> <span class="nw">(ทุกแหล่ง)</span>',
             lr["recall"]["mean"], lr["false_alarm_rate"]["mean"])]
    for origin, name in ORIGIN_TH.items():
        r = s["by_origin"][origin]["Logistic Regression"]
        rows.append((f'<span class="nw">ข้ามแหล่ง:</span> <span class="nw">{name}</span>',
                     r["recall"], r["false_alarm_rate"]))
    table = "".join(
        f'<tr><td>{name}</td><td>{rec:.0%}</td><td>{fa:.0%}</td>'
        f'<td class="{"pass" if rec >= GOAL_RECALL and fa <= GOAL_FALSE_ALARM else "miss"}">'
        f'{"ผ่าน" if rec >= GOAL_RECALL and fa <= GOAL_FALSE_ALARM else "ยังไม่ถึง"}</td></tr>' for name, rec, fa in rows)
    real = s["data"]["real"]
    return (box("สุ่มแบ่ง 10 รอบ (ข้อความจริง 25% เป็นชุดทดสอบ)", f'<div class="kpis">{kpis}</div>')
            + box(f"เทียบกับเป้าหมาย (จับได้ ≥ {GOAL_RECALL:.0%} · เตือนผิด ≤ {GOAL_FALSE_ALARM:.0%})",
                  '<table><tr><th>ชุดทดสอบ</th><th>จับได้</th><th>เตือนผิด</th><th>ผล</th></tr>' + table + '</table>'
                  '<p class="note">ข้ามแหล่ง = train โดยไม่ใช้ข้อมูลจากแหล่งนั้น แล้วทดสอบกับแหล่งนั้น</p>', "")
            + box("โมเดลและข้อมูล",
                  f'<p style="margin:0">TF-IDF (คำเดี่ยว + คู่คำ) → Logistic Regression · ข้อความจริง '
                  f'หลอก {real.get("scam", 0):,} · ปกติ {real.get("normal", 0):,}</p>'))


st.markdown('<div class="hero"><p class="kicker">MINI PROJECT · 240-318 AI-ML</p><h1>ScamSMS</h1>'
            '<p class="sub">เช็ค SMS มิจฉาชีพภาษาไทย ด้วย <span class="nw">Machine Learning</span></p></div>', unsafe_allow_html=True)

pipe = load_model()
if BOOTH:
    tab_check, tab_about = st.tabs(["ตรวจข้อความ", "เกี่ยวกับโมเดล"])
    tab_collect = None
else:
    tab_check, tab_collect, tab_about = st.tabs(["ตรวจข้อความ", "เก็บข้อมูล", "เกี่ยวกับโมเดล"])

with tab_check:
    if pipe is None:
        st.error("ยังไม่มี model.joblib - รัน `python train.py` ก่อน")
    else:
        st.markdown('<p class="label">ลองตัวอย่าง SMS จริง</p>', unsafe_allow_html=True)
        st.caption("แต่ละหัวข้อมีของจริง 1 ของหลอก 1 ลองทายก่อนว่าอันไหนหลอก")
        cols = st.container(key="examples").columns(4)
        for i, (name, example) in enumerate(EXAMPLES.items()):
            cols[i % 4].button(name, key=f"example_{i}", on_click=use_example, args=(example,),
                               width="stretch")

        txt = st.text_area("วาง SMS ที่ได้รับ (ออกแบบมาสำหรับ SMS ไม่ใช่แชทส่วนตัว)", key="msg", height=140,
                           placeholder="เช่น พัสดุของท่านถูกกักไว้ที่ศุลกากร กรุณาชำระค่าธรรมเนียม...")
        clicked = st.button("ตรวจข้อความ", type="primary", width="stretch")
        auto = st.session_state.pop("auto_check", False)
        checked = (clicked or auto) and txt.strip()
        if checked and pipe.named_steps["tfidf"].transform([txt]).nnz == 0:
            # No known word at all: the score would just be the class prior (~50%), not a judgement.
            st.info("โมเดลไม่รู้จักคำในข้อความนี้เลย ตัดสินไม่ได้ - ลองวาง SMS ทั้งข้อความ")
        elif checked:
            p = float(pipe.predict_proba([txt])[0][1])
            st.session_state["last"] = {"text": txt, "p": p}
            st.markdown(verdict_box(p), unsafe_allow_html=True)
            st.markdown(reasons_box(txt, contributions(pipe, txt)), unsafe_allow_html=True)
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
    if SUMMARY.exists():
        st.markdown(about_html(json.loads(SUMMARY.read_text(encoding="utf-8"))), unsafe_allow_html=True)
    elif METRICS.exists():
        m = json.loads(METRICS.read_text(encoding="utf-8"))
        st.write(f"**โมเดล:** {m['model']} · ข้อมูล {m['n_total']} ข้อความ")
        st.caption("รัน `python evaluate.py` เพื่อดูผลประเมินแบบเต็ม")
    else:
        st.info("รัน `python train.py` ก่อน")
