#!/usr/bin/env python3
"""Academic poster v4 — direct top-down draw, no cards, tight packing."""

import os
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FONT_DIR = r"C:\Users\creegon\AppData\Roaming\Claude\local-agent-mode-sessions\skills-plugin\ba5a2ee2-f0a1-4868-be93-b045b48145be\bc18f87b-9577-417f-ba3f-544654b310ef\skills\canvas-design\canvas-fonts"
OUT = r"C:\Users\creegon\Desktop\Design_Project\website"

for n, f in [("O","Outfit-Regular.ttf"),("OB","Outfit-Bold.ttf"),("C","CrimsonPro-Regular.ttf"),
             ("CB","CrimsonPro-Bold.ttf"),("M","GeistMono-Regular.ttf"),("MB","GeistMono-Bold.ttf")]:
    p = os.path.join(FONT_DIR, f)
    if os.path.exists(p): pdfmetrics.registerFont(TTFont(n, p))

# Colors
CR=HexColor("#B31B1B"); CD=HexColor("#8C1515"); BL=HexColor("#1565C0"); BLL=HexColor("#E3F2FD")
PK=HexColor("#880E4F"); PKL=HexColor("#FCE4EC"); GN=HexColor("#1B5E20"); GNL=HexColor("#E8F5E9")
GY=HexColor("#ECEEF0"); BG=HexColor("#FFFFFF"); BD=HexColor("#D0D0D0"); TX=HexColor("#1a1a1a")
T2=HexColor("#333"); MU=HexColor("#666"); RD=HexColor("#C62828"); YH=HexColor("#FFFDE7")

W, H = 48*inch, 36*inch

def rr(cv,x,y,w,h,r=8,fl=None,st=None,sw=1.5):
    cv.saveState()
    if fl: cv.setFillColor(fl)
    if st: cv.setStrokeColor(st); cv.setLineWidth(sw)
    p=cv.beginPath(); p.roundRect(x,y,w,h,r)
    cv.drawPath(p,fill=bool(fl),stroke=bool(st)); cv.restoreState()

def dt(cv,x,y,s,f="C",sz=20,co=TX,al="left",mw=None):
    cv.saveState(); cv.setFont(f,sz); cv.setFillColor(co)
    tw=cv.stringWidth(s,f,sz)
    if al=="center" and mw: x+=(mw-tw)/2
    elif al=="right" and mw: x+=mw-tw
    cv.drawString(x,y,s); cv.restoreState(); return tw

def dw(cv,x,y,s,f="C",sz=24,co=T2,mw=500,ld=1.35):
    words=s.split(); line=""; lh=sz*ld; cy=y
    for w in words:
        t=line+(" " if line else "")+w
        if cv.stringWidth(t,f,sz)>mw and line:
            dt(cv,x,cy,line,f,sz,co); cy-=lh; line=w
        else: line=t
    if line: dt(cv,x,cy,line,f,sz,co); cy-=lh
    return cy

def shdr(cv,x,y,title,w,badge=None,bc=CR):
    if badge:
        bw=cv.stringWidth(badge,"OB",20)+24
        rr(cv,x,y-7,bw,34,r=17,fl=bc); dt(cv,x+12,y,badge,"OB",20,white)
        dt(cv,x+bw+12,y,title,"OB",34,CD)
    else:
        dt(cv,x,y,title,"OB",34,CD)
    y-=16; cv.saveState(); cv.setStrokeColor(CR); cv.setLineWidth(3.5)
    cv.line(x,y,x+w,y); cv.restoreState(); return y-12

def tbl(cv,x,y,hds,rows,cws,w,hl=None,fs=22):
    hl=hl or set(); rh=fs*2.0
    rr(cv,x,y-rh+4,w,rh,r=5,fl=GY)
    cx=x+10
    for h,cw in zip(hds,cws): dt(cv,cx,y-rh+int(fs*.68),h,"OB",fs-1,TX,mw=cw); cx+=cw
    y-=rh; cv.saveState(); cv.setStrokeColor(BD); cv.setLineWidth(2); cv.line(x,y+4,x+w,y+4); cv.restoreState()
    for ri,row in enumerate(rows):
        if ri in hl: rr(cv,x,y-rh+4,w,rh,r=3,fl=YH)
        cx=x+10
        for cell,cw in zip(row,cws):
            if isinstance(cell,tuple): s,co=cell[0],cell[1]; fn=cell[2] if len(cell)>2 else "C"
            else: s,co,fn=str(cell),T2,"C"
            dt(cv,cx,y-rh+int(fs*.68),s,fn,fs-1,co,mw=cw); cx+=cw
        y-=rh
        cv.saveState(); cv.setStrokeColor(HexColor("#DDD")); cv.setLineWidth(.5)
        cv.line(x+4,y+4,x+w-4,y+4); cv.restoreState()
    return y

def sbox(cv,x,y,num,lab,w,col=CR):
    h=100; rr(cv,x,y-h,w,h,r=8,fl=GY,st=BD,sw=.5)
    dt(cv,x,y-48,str(num),"OB",52,col,al="center",mw=w)
    dt(cv,x,y-78,lab,"C",17,MU,al="center",mw=w)
    return y-h-6

def cbox(cv,x,y,title,items,w,ac,bg):
    ih=30; h=48+len(items)*ih+10
    rr(cv,x,y-h,w,h,r=8,fl=bg)
    cv.saveState(); cv.setFillColor(ac); cv.rect(x,y-h+8,5,h-16,fill=1,stroke=0); cv.restoreState()
    dt(cv,x+16,y-32,title,"OB",24,ac)
    iy=y-60
    for it in items: dt(cv,x+20,iy,"• "+it,"C",21,T2); iy-=ih
    return y-h-6

def sep(cv,x,y,w):
    """Section separator — subtle line."""
    y-=14; cv.saveState(); cv.setStrokeColor(HexColor("#CCC")); cv.setLineWidth(1)
    cv.setDash(3,3); cv.line(x,y,x+w,y); cv.restoreState()
    return y-18


# ══════════════════════════════════════
cv = canvas.Canvas(os.path.join(OUT,"poster.pdf"), pagesize=(W,H))
cv.setFillColor(BG); cv.rect(0,0,W,H,fill=1,stroke=0)

# ── HEADER ──
HH=2.1*inch
cv.setFillColor(CD); cv.rect(0,H-HH,W,HH,fill=1,stroke=0)
cv.setFillColor(CR); cv.rect(0,H-.3*inch,W,.3*inch,fill=1,stroke=0)
dt(cv,.4*inch,H-.82*inch,"Source Attribution of Watermarked Text","OB",76,white)
dt(cv,.4*inch,H-1.2*inch,"Han Li (hl2595)  ·  Kangbo Hao (kh873)  ·  Ruoxuan Cao (rc986)","O",30,HexColor("#FFCDD2"))
dt(cv,.4*inch,H-1.55*inch,"M.Eng. Design Project · ECE, Cornell · Advisor: Prof. Vikram Krishnamurthy · ECE 5995/5996 · 2025–2026","C",23,HexColor("#FFFFFF88"))
dt(cv,W-3.5*inch,H-.82*inch,"CORNELL","OB",58,HexColor("#FFFFFFAA"))
dt(cv,W-1.7*inch,H-1.25*inch,"ECE","O",38,HexColor("#FFFFFF66"))

# ── FOOTER ──
FH=.35*inch
cv.setFillColor(HexColor("#F5F5F5")); cv.rect(0,0,W,FH,fill=1,stroke=0)
dt(cv,.4*inch,.08*inch,"Cornell ECE 5995/5996 · 2025–2026  |  hl2595@cornell.edu  |  creegon.github.io/Design_Project","C",16,MU)

# ── Column setup ──
BT=H-HH-.08*inch; BB=FH+.08*inch
GAP=.22*inch
uw=W-.6*inch-2*GAP  # usable width (0.3in margin each side)
C1W=uw*(1/3.3); C2W=uw*(1.3/3.3); C3W=uw*(1/3.3)
C1X=.3*inch; C2X=C1X+C1W+GAP; C3X=C2X+C2W+GAP

# Draw light column backgrounds
for cx, cw in [(C1X,C1W),(C2X,C2W),(C3X,C3W)]:
    rr(cv, cx, BB, cw, BT-BB, r=12, fl=HexColor("#FAFBFC"), st=HexColor("#E0E2E5"), sw=1)

P = .2*inch  # padding inside columns

# ══════ COLUMN 1 ══════
y = BT - P

y = shdr(cv, C1X+P, y, "Motivation & Problem", C1W-2*P)
y = dw(cv, C1X+P, y, "Multiple LLMs embed cryptographic watermarks for provenance. When an observer receives text, which model produced it? Two attribution paradigms exist:", mw=C1W-2*P)
y -= 10
bw=(C1W-2*P-14)//2
y2=cbox(cv,C1X+P,y,"Watermark (Active)",["Per-model KGW key","z-test each key","Strong on clean text"],bw,BL,BLL)
cbox(cv,C1X+P+bw+14,y,"Fingerprint (Passive)",["TF-IDF n-gram features","4-way classifier","No key needed"],bw,PK,PKL)
y=y2-4
dt(cv,C1X+P,y,"Research Gaps","OB",28,TX); y-=34
for i,g in enumerate(["Does watermarking compress inter-model distributional differences?",
    "Head-to-head watermark vs. fingerprint attribution under attack?",
    "Can sentence-level watermarks (SimMark) complement token-level (KGW)?"]):
    dt(cv,C1X+P,y,f"{i+1}.","MB",22,CR); y=dw(cv,C1X+P+28,y,g,sz=22,mw=C1W-2*P-32); y-=6

y=sep(cv,C1X+P,y,C1W-2*P)

y = shdr(cv, C1X+P, y, "Method — KGW Watermarking", C1W-2*P)
y = dw(cv, C1X+P, y, "Kirchenbauer et al., ICML 2023. Token-level green/red partition with logit bias. Detection via z-test on green token count.", mw=C1W-2*P)
y -= 10
ew=C1W-2*P
rr(cv,C1X+P,y-48,ew,48,r=6,fl=GY)
dt(cv,C1X+P,y-34,"Z = (|s|_G − γT) / √(γ(1−γ)T)","M",26,TX,al="center",mw=ew)
y-=62
dt(cv,C1X+P,y,"Parameters:","CB",22,TX); y-=28
dt(cv,C1X+P,y,"γ = 0.25  ·  δ = 2.0  ·  z-threshold = 3.0","M",20,T2); y-=28
dt(cv,C1X+P,y,"Models:","CB",22,TX); y-=28
dt(cv,C1X+P,y,"LLaMA-3.2-1B, LLaMA-3.2-3B, Gemma-2B, Phi-2","C",21,T2); y-=28
dt(cv,C1X+P,y,"Attacks:","CB",22,TX); y-=28
dt(cv,C1X+P,y,"Synonym sub. 10–70%, Ins/Del 10–30%, Reorder, Neural (Qwen-3)","C",21,T2)

y=sep(cv,C1X+P,y,C1W-2*P)

y = shdr(cv, C1X+P, y, "Distributional Compression", C1W-2*P, badge="F1", bc=CR)
y = dw(cv, C1X+P, y, "Watermarking shrinks inter-model differences to 1/3 of clean baseline (Gemma-2B vs Phi-2).", mw=C1W-2*P)
y -= 8
tw=C1W-2*P; cws=[tw*.46,tw*.27,tw*.27]
y = tbl(cv,C1X+P,y,["Setting","Wt.Diff","Ent.Diff"],[
    ["Indep. WM keys","0.24","0.76"],
    [("No WM (clean)",GN,"CB"),("0.76",GN,"MB"),("2.30",GN,"MB")],
    [("Shared green list",GN,"CB"),("1.56",GN,"MB"),("4.48",GN,"MB")],
],cws,tw,fs=21)
y -= 8
y = dw(cv,C1X+P,y,"Green-list bias dominates each model's natural signature — a direct headwind for passive fingerprint attribution.",mw=C1W-2*P,sz=21,co=MU)


# ══════ COLUMN 2 ══════
y = BT - P

y = shdr(cv, C2X+P, y, "Robustness Crossover", C2W-2*P, badge="F2", bc=BL)
y = dw(cv, C2X+P, y, "Two paradigms fail in non-overlapping regimes. Watermark: cliff-like collapse. Fingerprint: graceful degradation. Lines cross at 50–70% synonym substitution.", mw=C2W-2*P, sz=22)
y -= 8
tw=C2W-2*P; cws=[tw*.26,tw*.22,tw*.22,tw*.30]
y = tbl(cv,C2X+P,y,["Synonym %","Watermark","Fingerprint","Winner"],[
    ["0% (clean)",("96.7%",GN,"MB"),"76.7%",("Watermark",BL)],
    ["10–30%",("93–97%",GN,"MB"),"70–77%",("Watermark",BL)],
    ["50%","73.3%","66.7%",("Watermark",BL)],
    [("70%",CD,"CB"),("13.3%",RD,"MB"),("63.3%",GN,"MB"),("Fingerprint",PK,"CB")],
],cws,tw,hl={3},fs=23)
y -= 12
sw=(C2W-2*P-24)//3
sbox(cv,C2X+P,y,"96.7%","WM clean accuracy",sw,GN)
sbox(cv,C2X+P+sw+12,y,"13.3%","WM @ 70% synonym",sw,RD)
y=sbox(cv,C2X+P+2*(sw+12),y,"63.3%","FP @ 70% synonym",sw,PK)
y -= 4
y = dw(cv,C2X+P,y,"Takeaway: production attribution should combine both — failure modes are non-overlapping.",mw=C2W-2*P,sz=22,co=TX,f="CB")

y=sep(cv,C2X+P,y,C2W-2*P)

y = shdr(cv, C2X+P, y, "Hybrid KGW + SimMark", C2W-2*P, badge="F3", bc=GN)
y = dw(cv, C2X+P, y, "SimMark (EMNLP 2025): sentence-level cosine similarity watermark. Combined with KGW via rejection sampling. Detection: OR logic — either positive → watermarked.", mw=C2W-2*P, sz=21)
y -= 10
dt(cv,C2X+P,y,"SimMark vs KGW — Moderate Paraphrase (5 rounds)","OB",24,TX); y-=8
tw=C2W-2*P; cws=[tw*.38,tw*.31,tw*.31]
y = tbl(cv,C2X+P,y,["Metric","SimMark","KGW"],[
    ["Survival Rate",("80%",GN,"MB"),("20%",RD,"MB")],
    ["z-score Decay","34%",("75%",RD,"MB")],
],cws,tw,fs=23)
y -= 14
dt(cv,C2X+P,y,"Hybrid Robustness (OR-logic detection)","OB",24,TX); y-=8
cws=[tw*.30,tw*.18,tw*.22,tw*.30]
y = tbl(cv,C2X+P,y,["Scenario","KGW","SimMark","Hybrid OR"],[
    ["Short, no atk",("66.7%",GN),"33.3%","66.7%"],
    [("Poetry, no atk",TX,"CB"),"66.7%","66.7%",("100%  ★",GN,"MB")],
    ["Long, mod. atk",("0%",RD),("100%",GN,"MB"),("100%",GN,"MB")],
    ["Any, aggressive",("0%",RD),("0%",RD),("0%",RD)],
],cws,tw,hl={1},fs=23)
y -= 14
bw=(C2W-2*P-14)//2
y2=cbox(cv,C2X+P,y,"KGW Rescues",["Short text (≤3 sent.): SimMark lacks stats","Haiku: KGW z=5.89, SimMark z=−0.58"],bw,BL,BLL)
cbox(cv,C2X+P+bw+14,y,"SimMark Rescues",["Moderate rewrite: semantics preserved","Long: SimMark z=3.87, KGW z=1.66"],bw,PK,PKL)
y=y2


# ══════ COLUMN 3 ══════
y = BT - P

y = shdr(cv, C3X+P, y, "Multi-LLM Chain Paraphrase", C3W-2*P, badge="F4", bc=PK)
y = dw(cv, C3X+P, y, "17 experiments: Llama-3.2-3B → Qwen-3 paraphraser. KGW watermarks do not survive neural paraphrase.", mw=C3W-2*P, sz=21)
y -= 8
sw=(C3W-2*P-24)//3
sbox(cv,C3X+P,y,"17","chain experiments",sw,TX)
sbox(cv,C3X+P+sw+12,y,"6%","survival (1/17)",sw,RD)
y=sbox(cv,C3X+P+2*(sw+12),y,"5.07","avg z-decay",sw,PK)
y -= 6
tw=C3W-2*P; cws=[tw*.34,tw*.18,tw*.24,tw*.24]
y = tbl(cv,C3X+P,y,["Mode","Orig z","After z","Survived"],[
    ["Standard","5.73",("−1.16",RD),("No",RD)],
    ["Creative","5.73","0.77",("No",RD)],
    ["Structure","5.73","0.05",("No",RD)],
    ["Summary-exp.","5.73",("−0.20",RD),("No",RD)],
],cws,tw,fs=21)
y -= 6
y = dw(cv,C3X+P,y,"Neural paraphrase is far more destructive than symbolic synonym substitution.",mw=C3W-2*P,sz=19,co=MU)

y=sep(cv,C3X+P,y,C3W-2*P)

y = shdr(cv, C3X+P, y, "Conclusions", C3W-2*P)
for num,t in [
    ("1.","Watermarking compresses distributions — makes different models look more alike."),
    ("2.","Watermark & fingerprint: non-overlapping failure modes (crossover at 50–70%)."),
    ("3.","KGW + SimMark are complementary — Hybrid OR achieves 100% on poetry."),
    ("4.","Neural paraphrase defeats KGW alone — only 1/17 survived."),
    ("5.","Aggressive attack remains an open challenge for all current methods."),
]:
    dt(cv,C3X+P,y,num,"MB",22,CR); y=dw(cv,C3X+P+30,y,t,sz=22,mw=C3W-2*P-34); y-=8

y=sep(cv,C3X+P,y,C3W-2*P)

y = shdr(cv, C3X+P, y, "Future Work", C3W-2*P)
for f in ["Hybrid attribution classifier: KGW z-scores + TF-IDF features in one model",
    "Neural paraphrase attacks: extend with GPT/Claude rewrite",
    "Efficient hybrid generation: sentence-level beam search (SentBS)",
    "Adaptive SimMark intervals auto-tuned per text type"]:
    dt(cv,C3X+P,y,"→","M",21,CR); y=dw(cv,C3X+P+26,y,f,sz=22,mw=C3W-2*P-30); y-=6

y=sep(cv,C3X+P,y,C3W-2*P)

dt(cv,C3X+P,y,"References","OB",28,TX); y-=10
cv.saveState(); cv.setStrokeColor(BD); cv.setLineWidth(1); cv.line(C3X+P,y,C3X+C3W-P,y); cv.restoreState(); y-=24
for ref in [
    "[1] Kirchenbauer et al. A Watermark for LLMs. ICML 2023. arXiv:2301.10226",
    "[2] Aghdam et al. SimMark: Sentence-Level Watermarking. EMNLP 2025",
    "[3] Pang et al. No Free Lunch in LLM Watermarking. 2024",
    "[4] Liu et al. Watermark Stealing in LLMs. 2024",
    "[5] Dathathri et al. SynthID-Text. Nature 2024",
]:
    dt(cv,C3X+P,y,ref,"C",19,MU); y-=28
y-=10
dt(cv,C3X+P,y,"Website: creegon.github.io/Design_Project","M",20,CR); y-=28
dt(cv,C3X+P,y,"Code: github.com/creegon/Design_Project","M",20,CR)


cv.save()
print("✅ PDF")

try:
    import fitz
    doc=fitz.open(os.path.join(OUT,"poster.pdf"))
    pix=doc[0].get_pixmap(dpi=150)
    pix.save(os.path.join(OUT,"poster.png")); print(f"✅ PNG {pix.width}x{pix.height}"); doc.close()
except Exception as e: print(f"⚠ {e}")
