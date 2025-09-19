
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Takaisinmaksu, NPV & Kassavirta", page_icon="💶", layout="wide")

st.title("💶 Takaisinmaksu, NPV & Kassavirta – laskuri")

with st.sidebar:
    st.header("Perusparametrit")
    invest = st.number_input("Investointi (€)", min_value=0, value=600_000, step=10_000, format="%d")
    lifetime = st.number_input("Tekninen elinkaari (v)", min_value=1, value=25, step=1)
    annual_saving = st.number_input("Vuosisäästö ensimmäisenä vuotena (€ / v)", min_value=0, value=50_000, step=1_000, format="%d")
    saving_growth = st.number_input("Säästön kasvu (% vuodessa)", min_value=-50.0, max_value=50.0, value=0.0, step=0.1) / 100.0

    st.header("Rahoitus")
    financed_share = st.slider("Lainalla katettava osuus investoinnista (%)", 0, 100, 100) / 100.0
    loan_years = st.number_input("Laina-aika (v)", min_value=1, value=20, step=1)
    interest = st.number_input("Nimelliskorko (% p.a.)", min_value=0.0, value=4.0, step=0.1) / 100.0
    annuity = st.toggle("Annuiteettilaina (tasamaksu)", value=True)

    st.header("NPV")
    discount_rate = st.number_input("Diskonttokorko (% p.a.)", min_value=0.0, value=4.0, step=0.1) / 100.0

    st.header("Ylläpito ja korvausinvestoinnit (valinnaiset)")
    om_fixed = st.number_input("Vuosittainen ylläpitokulu (€ / v)", min_value=0, value=0, step=500, format="%d")
    repl_interval = st.number_input("Korvausinvestoinnin väli (v, 0 = ei)", min_value=0, value=0, step=1)
    repl_cost = st.number_input("Korvausinvestoinnin kustannus (€)", min_value=0, value=0, step=10_000, format="%d")

# --- apufunktiot ---
def annuity_payment(principal, r, n):
    if principal <= 0:
        return 0.0
    if r == 0:
        return principal / n
    return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

def linear_payment(principal, r, n, year):
    """Tasalyhennys: korko laskee vuosittain. year alkaa 1..n"""
    if principal <= 0:
        return 0.0, 0.0, 0.0
    principal_payment = principal / n
    remaining = principal - (year - 1) * principal_payment
    interest_payment = remaining * r
    total = principal_payment + interest_payment
    return total, principal_payment, interest_payment

def payback_year(cum_series, invest_amount):
    """Ensimmäinen vuosi jolloin kumulatiivinen kassavirta >= investointi (ei diskontattu). Palauttaa desimaalivuoden."""
    c = cum_series.values
    for i in range(len(c)):
        if c[i] >= invest_amount:
            if i == 0:
                return 1.0
            # interpolaatiota edellisen vuoden ja tämän välillä
            prev = c[i-1]
            need = invest_amount - prev
            delta = c[i] - prev
            frac = 0.0 if delta == 0 else need / delta
            return (i) + frac  # i on 1-indeksoituna kuluva vuosi
    return math.inf

# --- laskenta ---
n_years = int(lifetime)

# säästöt polkuna
savings = np.array([(annual_saving * ((1 + saving_growth) ** (t))) for t in range(n_years)], dtype=float)

# rahoitus
loan_principal = invest * financed_share
own_cash = invest * (1 - financed_share)

payments = np.zeros(n_years)
interests = np.zeros(n_years)
principals = np.zeros(n_years)

if annuity:
    pay = annuity_payment(loan_principal, interest, loan_years)
    for y in range(n_years):
        if y < loan_years:
            # hajotetaan karkeasti korko/lyhennys
            # Annuiteetin korko osuus: r * remaining, jossa remaining approksimoidaan rekursiivisesti
            if y == 0:
                remaining = loan_principal
            else:
                remaining = remaining * (1 + interest) - pay
            interest_part = remaining * interest
            principal_part = pay - interest_part
            payments[y] = pay
            interests[y] = max(0.0, interest_part)
            principals[y] = max(0.0, principal_part)
else:
    for y in range(n_years):
        if y < loan_years:
            total, p_part, i_part = linear_payment(loan_principal, interest, loan_years, y+1)
            payments[y] = total
            interests[y] = i_part
            principals[y] = p_part

# ylläpito ja korvausinvestoinnit
om = np.full(n_years, float(om_fixed))
replacements = np.zeros(n_years)
if repl_interval and repl_cost and repl_interval > 0:
    for y in range(repl_interval, n_years, repl_interval):
        replacements[y] = repl_cost

# vuosikassavirta (positiivinen = hyöty)
cashflow = savings - payments - om - replacements
# investointikustannus vuonna 0 (oman rahan osuus ulos) + lainan nosto sisään, mutta tässä mallissa
# tarkastellaan kassavirtaa "säästö - lainanhoito - kulut", ja investointi huomioidaan erikseen NPV:ssä
cum_cash = np.cumsum(cashflow)

# takaisinmaksuaika (ei diskontattu)
pb = payback_year(cum_cash, invest)

# NPV (oletetaan investointi tapahtuu vuonna 0 täysimääräisenä)
years = np.arange(1, n_years+1)
discount_factors = 1 / ((1 + discount_rate) ** years)
npv = -invest + np.sum(cashflow * discount_factors)

# DataFrame raportointiin
df = pd.DataFrame({
    "Vuosi": years,
    "Säästö (€)": savings.round(2),
    "Lainan maksut (€)": payments.round(2),
    "Korko (€)": interests.round(2),
    "Lyhennys (€)": principals.round(2),
    "Ylläpito (€)": om.round(2),
    "Korvausinvestointi (€)": replacements.round(2),
    "Nettokassavirta (€)": cashflow.round(2),
    "Kumulatiivinen kassavirta (€)": cum_cash.round(2),
    "Diskonttaustekijä": discount_factors.round(5),
    "Diskontattu kassavirta (€)": (cashflow * discount_factors).round(2),
})

# --- UI ---
col1, col2, col3 = st.columns(3)
col1.metric("NPV (€/elinkaari)", f"{npv:,.0f} €")
col2.metric("Takaisinmaksuaika (ei diskontattu)", "∞" if math.isinf(pb) else f"{pb:.1f} vuotta")
col3.metric("Ensimmäisen vuoden nettokassavirta", f"{cashflow[0]:,.0f} €")

st.divider()

st.subheader("Vuosikassavirta (€)")
fig = plt.figure()
plt.plot(df["Vuosi"], df["Nettokassavirta (€)"])
plt.xlabel("Vuosi")
plt.ylabel("€ / vuosi")
st.pyplot(fig, clear_figure=True)

st.subheader("Kumulatiivinen kassavirta (€)")
fig2 = plt.figure()
plt.plot(df["Vuosi"], df["Kumulatiivinen kassavirta (€)"])
plt.axhline(invest, linestyle="--")
plt.text(1, invest * 1.02, "Investointi", va="bottom")
plt.xlabel("Vuosi")
plt.ylabel("€")
st.pyplot(fig2, clear_figure=True)

st.subheader("Diskontattu kassavirta ja NPV")
fig3 = plt.figure()
plt.bar(df["Vuosi"], df["Diskontattu kassavirta (€)"])
plt.xlabel("Vuosi")
plt.ylabel("€ (nykyarvo)")
st.pyplot(fig3, clear_figure=True)

st.divider()
st.subheader("Taulukko")
st.caption("Lataa CSV oikeasta yläkulmasta (kolmen pisteen valikko).")
st.dataframe(df, use_container_width=True)

with st.expander("Laskentalogiikka (tiivistelmä)"):
    st.markdown("""
- **Nettokassavirta** = säästö − lainanhoito − ylläpito − korvausinvestoinnit  
- **Kumulatiivinen kassavirta** = vuosikassavirtojen summa  
- **Takaisinmaksuaika** = ensimmäinen vuosi, jolloin kumulatiivinen kassavirta ≥ investointi (ei diskontattu)  
- **NPV** = −investointi + Σ (nettovuosikassavirta / (1 + diskonttokorko)^vuosi)  
- Lainoitus: valittavissa annuiteetti tai tasalyhennys.
""")
