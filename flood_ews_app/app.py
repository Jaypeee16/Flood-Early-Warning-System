import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import date

from utils.prediction import (
    predict_flood_probability, get_city_info, sigmoid_curve_data,
    is_using_fallback, get_optimal_threshold, get_model_features,
)
from utils.warning_tiers import get_tier

st.set_page_config(
    page_title="Flood EWS — Metro Manila",
    page_icon="🌧️", layout="wide",
    initial_sidebar_state="expanded",
)

CITIES            = ["Manila", "Marikina", "Quezon City", "Pasig"]
OPTIMAL_THRESHOLD = get_optimal_threshold()

TIERS = {
    "Green":  dict(g1="22,197,94",   g2="5,150,105",   ring="#22c55e", text="#14532d", bg="#f0fdf4", border="#bbf7d0", badge="#dcfce7"),
    "Yellow": dict(g1="251,191,36",  g2="245,130,31",   ring="#f59e0b", text="#713f12", bg="#fffbeb", border="#fde68a", badge="#fef9c3"),
    "Orange": dict(g1="249,115,22",  g2="239,68,68",    ring="#f97316", text="#7c2d12", bg="#fff7ed", border="#fed7aa", badge="#ffedd5"),
    "Red":    dict(g1="239,68,68",   g2="185,28,28",    ring="#ef4444", text="#7f1d1d", bg="#fef2f2", border="#fecaca", badge="#fee2e2"),
}

def h(tag, content="", **kw):
    """Tiny helper — build a single HTML element with inline styles from kwargs."""
    style = ";".join(f"{k.replace('_','-')}:{v}" for k, v in kw.items())
    return f"<{tag} style='{style}'>{content}</{tag}>"

def metric_card(label, value, sub, g1, g2, bar_pct=None, icon=""):
    """Return a self-contained gradient metric card as a single HTML string."""
    bar = ""
    if bar_pct is not None:
        fill_w = f"{min(bar_pct, 100):.1f}%"
        bar = (f"<div style='background:rgba(255,255,255,0.22);border-radius:99px;"
               f"height:5px;overflow:hidden;margin-top:10px;'>"
               f"<div style='height:5px;border-radius:99px;"
               f"background:rgba(255,255,255,0.88);width:{fill_w};'></div></div>")
    orb = ("<div style='position:absolute;width:110px;height:110px;border-radius:50%;"
           "background:rgba(255,255,255,0.1);top:-28px;right:-24px;pointer-events:none;'></div>"
           "<div style='position:absolute;width:65px;height:65px;border-radius:50%;"
           "background:rgba(255,255,255,0.07);bottom:-18px;right:28px;pointer-events:none;'></div>")
    lbl  = (f"<p style='font-size:0.67rem;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:0.1em;opacity:0.8;margin:0 0 6px;color:#fff;'>{label}</p>")
    val  = (f"<p style='font-size:2.25rem;font-weight:900;letter-spacing:-0.03em;"
            f"line-height:1;margin:0;color:#fff;'>{icon}{value}</p>")
    sub_ = (f"<p style='font-size:0.73rem;opacity:0.72;font-weight:500;"
            f"margin:5px 0 0;color:#fff;'>{sub}</p>")
    return (f"<div style='border-radius:22px;padding:1.3rem 1.5rem 1.2rem;color:#fff;"
            f"position:relative;overflow:hidden;"
            f"box-shadow:0 10px 30px rgba(0,0,0,0.15);height:148px;"
            f"background:linear-gradient(135deg,rgb({g1}),rgb({g2}));'>"
            f"{orb}{lbl}{val}{sub_}{bar}</div>")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, sans-serif;
    background: #f0f4f8;
}
#MainMenu, footer { visibility: hidden; }

/* Sidebar shell */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e8edf2;
}
/* Sidebar labels */
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider   label,
[data-testid="stSidebar"] .stDateInput label {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #94a3b8 !important;
}
/* Sidebar select + date inputs — force white bg / dark text */
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"]  > div,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] [role="combobox"] {
    background-color: #f8fafc !important;
    color: #1e293b !important;
    border-color: #e2e8f0 !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg { color: #64748b !important; }
/* Date input specifically */
[data-testid="stSidebar"] [data-testid="stDateInput"] > div,
[data-testid="stSidebar"] [data-testid="stDateInput"] input,
[data-testid="stSidebar"] [data-testid="stDateInput"] [data-baseweb="input"] {
    background-color: #f8fafc !important;
    background: #f8fafc !important;
    color: #1e293b !important;
    border-color: #e2e8f0 !important;
    border-radius: 10px !important;
}
/* Slot wrapper inside date input */
[data-testid="stSidebar"] [data-testid="stDateInput"] div[class*="InputContainer"],
[data-testid="stSidebar"] [data-testid="stDateInput"] div[class*="Input"] {
    background-color: #f8fafc !important;
    color: #1e293b !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px; background: #e2e8f0;
    border-radius: 12px; padding: 3px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px; padding: 0.42rem 1.1rem;
    font-size: 0.84rem; font-weight: 600;
    color: #64748b; background: transparent;
}
.stTabs [aria-selected="true"] {
    background: #fff !important;
    color: #1e293b !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
uses_antecedent = "antecedent_3day" in get_model_features()

with st.sidebar:
    st.markdown(
        "<div style='background:linear-gradient(135deg,#4f46e5,#7c3aed);"
        "border-radius:18px;padding:1.15rem 1.25rem;margin-bottom:1.4rem;"
        "box-shadow:0 10px 28px rgba(79,70,229,0.35);'>"
        "<div style='font-size:1.3rem;line-height:1;'>🌧️</div>"
        "<div style='color:#fff;font-size:1.05rem;font-weight:800;margin-top:0.4rem;'>Flood EWS</div>"
        "<div style='color:#c4b5fd;font-size:0.73rem;font-weight:500;margin-top:0.2rem;'>Metro Manila</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    city     = st.selectbox("City", CITIES, index=1)
    rainfall = st.slider("Rainfall (mm / day)", 0.0, 100.0, 0.0, 0.5, format="%.1f mm")

    if uses_antecedent:
        antecedent = st.slider(
            "Antecedent 3-day Rainfall (mm)", 0.0, 300.0,
            value=round(rainfall * 2, 1), step=1.0, format="%.0f mm",
            help="Total rainfall over the past 3 days.",
        )
    else:
        antecedent = 0.0

    forecast_date = st.date_input("Forecast Date", value=date.today())
    st.divider()

    if is_using_fallback():
        st.markdown(
            "<div style='background:#fef9c3;border:1px solid #fde68a;border-radius:12px;"
            "padding:0.65rem 0.9rem;display:flex;align-items:center;gap:0.6rem;'>"
            "<span style='font-size:1rem;'>⚠️</span>"
            "<span style='font-size:0.78rem;font-weight:600;color:#854d0e;line-height:1.4;'>"
            "Sigmoid fallback<br><span style='font-weight:400;opacity:0.75;'>No model file found</span>"
            "</span></div>",
            unsafe_allow_html=True,
        )
    else:
        feats = get_model_features()
        feat_pills = " ".join(
            f"<span style='background:#d1fae5;color:#065f46;border-radius:6px;"
            f"padding:0.1rem 0.45rem;font-size:0.7rem;font-weight:700;'>{f}</span>"
            for f in feats
        )
        st.markdown(
            f"<div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;"
            f"padding:0.65rem 0.9rem;display:flex;align-items:flex-start;gap:0.6rem;'>"
            f"<span style='font-size:1rem;margin-top:1px;'>✅</span>"
            f"<div>"
            f"<span style='font-size:0.78rem;font-weight:700;color:#14532d;'>Trained model loaded</span><br>"
            f"<div style='margin-top:0.3rem;display:flex;gap:0.3rem;flex-wrap:wrap;'>{feat_pills}</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

# ── Compute ───────────────────────────────────────────────────────────────────
probability = predict_flood_probability(rainfall, city, antecedent)
tier        = get_tier(probability)
city_info   = get_city_info(city)
prob_pct    = probability * 100
tn          = tier["tier_name"]
T           = TIERS[tn]
adj         = city_info["adjustment_factor"]
adj_pct     = (adj - 1) * 100
sign        = "+" if adj_pct >= 0 else ""

# Radial SVG gauge
CIRC = 2 * math.pi * 48
dash = probability * CIRC
svg_gauge = (
    f"<svg width='120' height='120' viewBox='0 0 120 120' "
    f"style='flex-shrink:0;filter:drop-shadow(0 4px 10px rgba(0,0,0,0.10));'>"
    f"<circle cx='60' cy='60' r='48' fill='none' stroke='#e8edf2' stroke-width='10'/>"
    f"<circle cx='60' cy='60' r='48' fill='none' stroke='{T['ring']}' stroke-width='10' "
    f"stroke-linecap='round' stroke-dasharray='{dash:.2f} {CIRC:.2f}' "
    f"transform='rotate(-90 60 60)'/>"
    f"<text x='60' y='56' text-anchor='middle' font-family='Inter,sans-serif' "
    f"font-weight='900' font-size='16' fill='{T['text']}'>{prob_pct:.1f}%</text>"
    f"<text x='60' y='70' text-anchor='middle' font-family='Inter,sans-serif' "
    f"font-weight='600' font-size='8.5' fill='{T['text']}' opacity='0.6'>PROBABILITY</text>"
    f"</svg>"
)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    f"<div style='display:flex;align-items:center;justify-content:space-between;"
    f"flex-wrap:wrap;gap:0.5rem;padding:0.5rem 0 0.85rem;'>"
    f"<div>"
    f"<p style='font-size:1.5rem;font-weight:800;color:#1e293b;line-height:1.2;margin:0;'>"
    f"Flood Early-Warning System</p>"
    f"<p style='font-size:0.79rem;color:#94a3b8;font-weight:500;margin:0.2rem 0 0;'>"
    f"Metro Manila &nbsp;·&nbsp; Logistic Regression + PAGASA Tier Classification &nbsp;·&nbsp; "
    f"Trigger Threshold: <strong style='color:#64748b;'>{OPTIMAL_THRESHOLD} mm/day</strong></p>"
    f"</div>"
    f"<div style='background:#fff;border-radius:12px;padding:0.45rem 1rem;"
    f"box-shadow:0 2px 8px rgba(0,0,0,0.06);font-size:0.82rem;"
    f"color:#475569;font-weight:600;white-space:nowrap;'>"
    f"{forecast_date.strftime('%B %d, %Y')}</div></div>",
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["Live Simulation", "Historical Data"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Simulation
# ════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── 4 Metric cards ───────────────────────────────────────────────────
    mc1, mc2, mc3, mc4 = st.columns(4, gap="medium")
    mc1.markdown(metric_card("Flood Probability", f"{prob_pct:.1f}%",
        f"{tn} Warning Level", T["g1"], T["g2"], bar_pct=prob_pct),
        unsafe_allow_html=True)
    mc2.markdown(metric_card("PAGASA Tier", tn,
        "Warning classification", "99,102,241", "139,92,246",
        icon=f"{tier['icon']} "),
        unsafe_allow_html=True)
    mc3.markdown(metric_card("Rainfall Input", f"{rainfall:.1f}",
        f"mm / day · {city}", "14,165,233", "6,182,212", bar_pct=rainfall),
        unsafe_allow_html=True)
    mc4.markdown(metric_card("City Adjustment", f"{adj:.2f}×",
        f"{sign}{adj_pct:.0f}% vs baseline", "168,85,247", "236,72,153"),
        unsafe_allow_html=True)

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # ── Left / Right columns ─────────────────────────────────────────────
    left, right = st.columns([2, 3], gap="medium")

    # ── LEFT ─────────────────────────────────────────────────────────────
    with left:
        # Status gauge card
        st.markdown(
            f"<div style='background:{T['bg']};border:1.5px solid {T['border']};"
            f"border-radius:22px;padding:1.5rem 1.6rem;margin-bottom:0.85rem;"
            f"box-shadow:0 6px 24px rgba(0,0,0,0.07);"
            f"display:flex;align-items:center;gap:1.35rem;'>"
            f"{svg_gauge}"
            f"<div style='min-width:0;'>"
            f"<p style='font-size:0.63rem;font-weight:800;text-transform:uppercase;"
            f"letter-spacing:0.12em;color:{T['text']};opacity:0.6;margin:0 0 0.25rem;'>Current Status</p>"
            f"<p style='font-size:2.8rem;font-weight:900;letter-spacing:-0.04em;"
            f"line-height:1;color:{T['text']};margin:0;'>{prob_pct:.1f}%</p>"
            f"<p style='font-size:1.05rem;font-weight:800;color:{T['text']};margin:0.2rem 0 0;'>"
            f"{tier['icon']} {tn} Warning</p>"
            f"<p style='font-size:0.73rem;color:{T['text']};opacity:0.5;"
            f"margin:0.2rem 0 0;font-weight:500;'>{city} · {forecast_date.strftime('%b %d, %Y')}</p>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # Action card
        st.markdown(
            f"<div style='background:#fff;border-radius:18px;padding:1.05rem 1.25rem;"
            f"box-shadow:0 4px 16px rgba(0,0,0,0.05);margin-bottom:0.85rem;"
            f"border-left:4.5px solid {T['ring']};'>"
            f"<p style='font-size:0.63rem;font-weight:800;text-transform:uppercase;"
            f"letter-spacing:0.1em;color:#94a3b8;margin:0 0 0.4rem;'>Recommended Action</p>"
            f"<p style='font-size:0.88rem;color:#334155;line-height:1.6;"
            f"font-weight:500;margin:0;'>{tier['recommended_action']}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # City card
        st.markdown(
            f"<div style='background:#fff;border-radius:18px;padding:1rem 1.25rem;"
            f"box-shadow:0 4px 16px rgba(0,0,0,0.05);margin-bottom:0.85rem;"
            f"display:flex;align-items:flex-start;gap:0.85rem;'>"
            f"<div style='width:38px;height:38px;border-radius:10px;background:{T['badge']};"
            f"display:flex;align-items:center;justify-content:center;"
            f"font-size:1.1rem;flex-shrink:0;'>📍</div>"
            f"<div>"
            f"<p style='font-size:0.63rem;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:0.09em;color:#94a3b8;margin:0 0 0.2rem;'>City Profile</p>"
            f"<p style='font-size:0.91rem;font-weight:700;color:#1e293b;margin:0;'>"
            f"{city} &nbsp;·&nbsp; <span style='color:{T['ring']};font-weight:800;'>{adj:.2f}×</span></p>"
            f"<p style='font-size:0.79rem;color:#64748b;margin:0.15rem 0 0;"
            f"line-height:1.5;'>{city_info['description']}</p>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # Tier legend
        legend_rows = [
            ("Green",  "#22c55e", "< 25%",    "Normal monitoring"),
            ("Yellow", "#f59e0b", "25 – 49%", "Stay alert"),
            ("Orange", "#f97316", "50 – 74%", "Prepare to evacuate"),
            ("Red",    "#ef4444", "≥ 75%",    "Evacuate immediately"),
        ]
        rows_html = "".join(
            f"<div style='display:flex;align-items:center;gap:0.65rem;"
            f"padding:0.42rem 0.55rem;border-radius:10px;font-size:0.81rem;"
            f"margin-bottom:0.18rem;"
            f"{'background:'+TIERS[n]['badge']+';' if n==tn else ''}'>"
            f"<div style='width:9px;height:9px;border-radius:50%;background:{c};"
            f"flex-shrink:0;"
            f"{'box-shadow:0 0 0 3px '+TIERS[n]['border']+';' if n==tn else ''}'></div>"
            f"<span style='font-weight:{'800' if n==tn else '600'};color:#1e293b;min-width:52px;'>{n}</span>"
            f"<span style='color:#94a3b8;font-size:0.74rem;min-width:60px;'>{r}</span>"
            f"<span style='color:#64748b;font-size:0.78rem;'>{a}</span>"
            f"</div>"
            for n, c, r, a in legend_rows
        )
        st.markdown(
            f"<div style='background:#fff;border-radius:18px;padding:1rem 1.25rem;"
            f"box-shadow:0 4px 16px rgba(0,0,0,0.05);'>"
            f"<p style='font-size:0.63rem;font-weight:800;text-transform:uppercase;"
            f"letter-spacing:0.1em;color:#94a3b8;margin:0 0 0.65rem;'>Warning Tier Reference</p>"
            f"{rows_html}</div>",
            unsafe_allow_html=True,
        )

    # ── RIGHT: Chart ──────────────────────────────────────────────────────
    with right:
        # Chart title — plain markdown so it sits cleanly above the chart
        st.markdown(
            f"<div style='background:#fff;border-radius:22px 22px 0 0;"
            f"padding:1.3rem 1.5rem 0.9rem;"
            f"box-shadow:0 6px 24px rgba(0,0,0,0.07);'>"
            f"<p style='font-size:0.94rem;font-weight:800;color:#1e293b;margin:0;'>"
            f"Flood Probability vs. Rainfall</p>"
            f"<p style='font-size:0.75rem;color:#94a3b8;font-weight:500;margin:0.15rem 0 0;'>"
            f"{city} &nbsp;·&nbsp; Antecedent {antecedent:.0f} mm &nbsp;·&nbsp; "
            f"Trigger threshold {OPTIMAL_THRESHOLD} mm/day</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        x_vals, y_vals = sigmoid_curve_data(city, antecedent)
        fig = go.Figure()

        # Zone fills
        for y0, y1, fill in [
            (0.00, 0.25, "rgba(34,197,94,0.07)"),
            (0.25, 0.50, "rgba(245,158,11,0.09)"),
            (0.50, 0.75, "rgba(249,115,22,0.10)"),
            (0.75, 1.00, "rgba(239,68,68,0.10)"),
        ]:
            fig.add_hrect(y0=y0, y1=y1, fillcolor=fill, line_width=0, layer="below")

        # Zone labels — anchored inside plot
        for y_mid, label, color in [
            (0.125, "GREEN",  "#16a34a"),
            (0.375, "YELLOW", "#d97706"),
            (0.625, "ORANGE", "#ea580c"),
            (0.875, "RED",    "#dc2626"),
        ]:
            fig.add_annotation(
                x=97, y=y_mid, text=f"<b>{label}</b>",
                showarrow=False, xanchor="right", yanchor="middle",
                font=dict(size=9.5, color=color, family="Inter,sans-serif"),
                bgcolor="rgba(255,255,255,0.78)", borderpad=3,
            )

        # Tier boundary dashes
        for y_val, color in [(0.25, "#d97706"), (0.50, "#ea580c"), (0.75, "#dc2626")]:
            fig.add_hline(y=y_val, line_dash="dot", line_color=color,
                          line_width=1.5, layer="below")

        # Trigger threshold
        fig.add_vline(x=OPTIMAL_THRESHOLD, line_dash="dash",
                      line_color="#cbd5e1", line_width=1.8)
        fig.add_annotation(
            x=OPTIMAL_THRESHOLD + 0.8, y=1.07,
            text=f"<b>Trigger {OPTIMAL_THRESHOLD} mm</b>",
            showarrow=False, xanchor="left",
            font=dict(size=10, color="#94a3b8", family="Inter,sans-serif"),
        )

        # Area fill
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode="none", fill="tozeroy",
            fillcolor=f"rgba({T['g1']},0.10)",
            showlegend=False, hoverinfo="skip",
        ))

        # Probability curve
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode="lines",
            name=f"P(flood) — {city}",
            line=dict(color=T["ring"], width=3, shape="spline", smoothing=0.8),
            hovertemplate="Rainfall: <b>%{x} mm</b><br>Probability: <b>%{y:.1%}</b><extra></extra>",
        ))

        # Current input dot
        fig.add_trace(go.Scatter(
            x=[rainfall], y=[probability],
            mode="markers+text",
            name=f"Now: {prob_pct:.1f}%",
            text=[f"  {prob_pct:.1f}%"],
            textposition="middle right",
            textfont=dict(size=12, color=T["ring"], family="Inter,sans-serif"),
            marker=dict(color=T["ring"], size=16, symbol="circle",
                        line=dict(color="white", width=3)),
            hovertemplate=f"<b>{rainfall:.1f} mm → {prob_pct:.1f}%</b><extra></extra>",
        ))

        fig.update_layout(
            xaxis=dict(
                title=dict(text="Rainfall (mm / day)",
                           font=dict(size=11, color="#94a3b8")),
                range=[0, 100], gridcolor="#f1f5f9", zeroline=False,
                tickfont=dict(size=10.5, color="#94a3b8"), showline=False,
            ),
            yaxis=dict(
                title=dict(text="Flood Probability",
                           font=dict(size=11, color="#94a3b8")),
                range=[0, 1.12], tickformat=".0%",
                gridcolor="#f1f5f9", zeroline=False,
                tickfont=dict(size=10.5, color="#94a3b8"), showline=False,
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.18, xanchor="left", x=0,
                font=dict(size=11, color="#64748b"), bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(l=0, r=20, t=16, b=16),
            height=455,
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font=dict(family="Inter,sans-serif"),
            hoverlabel=dict(bgcolor="white", font_size=12,
                            font_family="Inter,sans-serif", bordercolor="#e2e8f0"),
        )

        # Render chart inside a bottom-rounded card matching the header above
        st.markdown(
            "<div style='background:#fff;border-radius:0 0 22px 22px;"
            "box-shadow:0 6px 24px rgba(0,0,0,0.07);"
            "padding:0 0.5rem 0.5rem;'>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Historical Data
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        "<p style='font-size:0.92rem;font-weight:700;color:#1e293b;margin:0.3rem 0 0.1rem;'>"
        "Metro Manila Flood Events — Historical Profile (2016–2020)</p>"
        "<p style='font-size:0.76rem;color:#94a3b8;font-weight:500;margin:0 0 1rem;'>"
        "Source: Metro Manila Flood Prediction 2016–2020 · Kaggle · CC BY 4.0</p>",
        unsafe_allow_html=True,
    )

    hist_csv = os.path.join(os.path.dirname(__file__), "data", "historical_data.csv")
    if os.path.exists(hist_csv):
        df_raw = pd.read_csv(hist_csv)
        # Render as clean HTML table even for the real CSV
        rows_raw = "".join(
            "<tr>" + "".join(
                f"<td style='padding:0.75rem 1rem;color:#1e293b;font-size:0.84rem;"
                f"border-bottom:1px solid #f1f5f9;font-weight:{'600' if i==0 else '400'};'>{v}</td>"
                for i, v in enumerate(row)
            ) + "</tr>"
            for _, row in df_raw.iterrows()
        )
        headers_raw = "".join(
            f"<th style='padding:0.65rem 1rem;text-align:left;font-size:0.68rem;"
            f"font-weight:700;text-transform:uppercase;letter-spacing:0.07em;"
            f"color:#64748b;border-bottom:2px solid #e2e8f0;white-space:nowrap;'>{c}</th>"
            for c in df_raw.columns
        )
        st.markdown(
            f"<div style='background:#fff;border-radius:18px;overflow:hidden;"
            f"box-shadow:0 4px 20px rgba(0,0,0,0.07);'>"
            f"<table style='width:100%;border-collapse:collapse;font-family:Inter,sans-serif;'>"
            f"<thead><tr style='background:#f8fafc;'>{headers_raw}</tr></thead>"
            f"<tbody>{rows_raw}</tbody></table></div>",
            unsafe_allow_html=True,
        )
    else:
        summary = pd.DataFrame({
            "Year":                   [2016, 2017, 2018, 2019, 2020],
            "Avg Rainfall (mm/day)":  [28.4, 31.2, 25.7, 33.1, 29.8],
            "Flood Events":           [47,   52,   39,   58,   44  ],
            "Worst-Affected City":    ["Marikina","Marikina","Pasig","Marikina","Manila"],
            "Peak Rainfall (mm/day)": [88.3, 94.1, 76.5, 101.2, 82.7],
        })

        # ── Full-width table ──────────────────────────────────────────────
        st.markdown(
            "<p style='font-size:0.65rem;font-weight:800;text-transform:uppercase;"
            "letter-spacing:0.1em;color:#94a3b8;margin-bottom:0.7rem;'>Annual Summary</p>",
            unsafe_allow_html=True,
        )

        max_events = max(summary["Flood Events"])
        def event_color(v):
            t = v / max_events
            return f"rgb({int(99+t*(220-99))},{int(102+t*(38-102))},{int(241+t*(38-241))})"

        col_labels = ["Year", "Avg Rainfall (mm/day)", "Flood Events",
                      "Worst-Affected City", "Peak Rainfall (mm/day)"]
        align      = ["left", "right", "center", "left", "right"]
        headers_html = "".join(
            f"<th style='padding:0.65rem 1.1rem;text-align:{align[i]};"
            f"font-size:0.67rem;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:0.07em;color:#64748b;"
            f"border-bottom:2px solid #e2e8f0;white-space:nowrap;'>{c}</th>"
            for i, c in enumerate(col_labels)
        )
        rows_html_tbl = ""
        for i, row in summary.iterrows():
            ec = event_color(row["Flood Events"])
            stripe = "background:#fafbfc;" if i % 2 == 1 else ""
            rows_html_tbl += (
                f"<tr style='{stripe}'>"
                f"<td style='padding:0.75rem 1.1rem;font-weight:700;color:#1e293b;"
                f"font-size:0.85rem;border-bottom:1px solid #f1f5f9;'>{int(row['Year'])}</td>"
                f"<td style='padding:0.75rem 1.1rem;text-align:right;color:#475569;"
                f"font-size:0.85rem;border-bottom:1px solid #f1f5f9;'>{row['Avg Rainfall (mm/day)']} mm</td>"
                f"<td style='padding:0.75rem 1.1rem;text-align:center;"
                f"font-size:0.85rem;border-bottom:1px solid #f1f5f9;'>"
                f"<span style='background:{ec};color:#fff;font-weight:700;"
                f"border-radius:8px;padding:0.18rem 0.65rem;font-size:0.82rem;'>"
                f"{int(row['Flood Events'])}</span></td>"
                f"<td style='padding:0.75rem 1.1rem;color:#475569;"
                f"font-size:0.85rem;border-bottom:1px solid #f1f5f9;'>{row['Worst-Affected City']}</td>"
                f"<td style='padding:0.75rem 1.1rem;text-align:right;color:#475569;"
                f"font-size:0.85rem;border-bottom:1px solid #f1f5f9;'>{row['Peak Rainfall (mm/day)']} mm</td>"
                f"</tr>"
            )
        st.markdown(
            f"<div style='background:#fff;border-radius:18px;overflow:hidden;"
            f"box-shadow:0 4px 20px rgba(0,0,0,0.07);margin-bottom:1.2rem;'>"
            f"<table style='width:100%;border-collapse:collapse;font-family:Inter,sans-serif;'>"
            f"<thead><tr style='background:#f8fafc;'>{headers_html}</tr></thead>"
            f"<tbody>{rows_html_tbl}</tbody></table></div>",
            unsafe_allow_html=True,
        )

        # ── Full-width chart below ─────────────────────────────────────────
        st.markdown(
            "<p style='font-size:0.65rem;font-weight:800;text-transform:uppercase;"
            "letter-spacing:0.1em;color:#94a3b8;margin-bottom:0.55rem;'>"
            "Flood Events & Average Rainfall by Year</p>",
            unsafe_allow_html=True,
        )
        fig_h = go.Figure()
        fig_h.add_trace(go.Bar(
            x=summary["Year"].astype(str), y=summary["Flood Events"],
            name="Flood Events",
            marker=dict(color="rgba(99,102,241,0.82)", line=dict(width=0), cornerradius=7),
            hovertemplate="Year: %{x}<br>Events: <b>%{y}</b><extra></extra>",
        ))
        fig_h.add_trace(go.Scatter(
            x=summary["Year"].astype(str), y=summary["Avg Rainfall (mm/day)"],
            name="Avg Rainfall (mm/day)", yaxis="y2",
            mode="lines+markers",
            line=dict(color="#06b6d4", width=2.5, shape="spline", smoothing=0.8),
            marker=dict(size=10, color="#06b6d4", line=dict(color="white", width=2.5)),
            hovertemplate="Year: %{x}<br>Avg Rainfall: <b>%{y} mm</b><extra></extra>",
        ))
        fig_h.update_layout(
            xaxis=dict(
                title=dict(text="Year", font=dict(size=12, color="#475569")),
                gridcolor="#f1f5f9", zeroline=False,
                tickfont=dict(size=12, color="#475569"), showline=False,
            ),
            yaxis=dict(
                title=dict(text="Flood Events", font=dict(size=12, color="#475569")),
                gridcolor="#f1f5f9", zeroline=False,
                tickfont=dict(size=12, color="#475569"), showline=False,
            ),
            yaxis2=dict(
                title=dict(text="Avg Rainfall (mm/day)", font=dict(size=12, color="#0891b2")),
                overlaying="y", side="right", showgrid=False,
                tickfont=dict(size=12, color="#0891b2"), zeroline=False,
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.18,
                font=dict(size=12, color="#475569"), bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(l=0, r=70, t=10, b=20),
            height=320, plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            bargap=0.38, font=dict(family="Inter,sans-serif"),
            hoverlabel=dict(bgcolor="white", font_size=12,
                            font_family="Inter,sans-serif", bordercolor="#e2e8f0"),
        )
        st.plotly_chart(fig_h, width="stretch")
