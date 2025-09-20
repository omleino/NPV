# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="NPV & Takaisinmaksu – kevyt", page_icon="💶", layout="wide")

st.title("💶 NPV & Takaisinmaksu – kevyt laskuri")

with st.sidebar:
    st.header("Perusparametrit")
    invest = st.number_input("Investointi (€)", min_value=0, value=600000, step=10000, format="%d")
    lifetime = st.number_input("Tekninen elinkaari (v)", min_value=1, value=25, step=1)
    annual_saving = st.number_input("Vuosisäästö 1. vuotena (€ / v)", min_value=0, value=50000, step=1000, format="%d")
    saving_growth = st.number_input("Säästön kasvu (% / v)", min_value=-50.0, max_value=50.0, value=0.0, step=0.1) / 100.0

    st.header("Rahoitus")
    financed_share = st.slider("Lainan osuus investoinnista (%)", 0, 100, 100) / 100.0
    loan_years = st.number_input("Laina-aika (v)", min_value=1, value=20, step=1)
    interest = st.number_input("Korko (% p.a.)", min_value=0.0, value=4.0, step=0.1) / 100.0
    annuity = st.toggle("Annuiteettilaina (tasamaksu)", value=True)

    st.header("NPV")
    discount_rate = st.number_input("Diskonttokorko (% p.a.)", min_value=0.0, value=4.0, step=0.1) / 100.0

    st.header("Ylläpito ja korvausinvestoinnit (valinnaiset)")
    om_fixed = st.number_input("Ylläpitokulu (€ / v)", min_value=0, value=0, step=500, format="%d")
    repl_interval = st.number_input("Korvausinvestoinnin väli (v, 0 = ei)", min_value=0, value=0, step=1)
    repl_cost = st.number_input("Korvausinvestoinnin kustannus (€)", min_value=0, value=0, step=10000, format="%d")

# ---- apufunktiot ----
def annuity_payment(principal, r, n):
    if principal <= 0:
        return 0.0
    if r == 0:
        return principal / n
    return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

def linear_payment(principal, r, n, year):
    if principal <= 0:
        return 0.0, 0.0, 0.0
    principal_payment = principal / n
    remaining = principal - (year - 1) * principal_payment
    interest_payment = remaining * r
    total = principal_payment + interest_payment
    return total, principal_payment, interest_payment

def payback_year(cum_series, invest_amount):
    try:
        import numpy as _np
        c = _np.asarray(cum_series, dtype=float).ravel()
    except Exception:
        try:
            c = [float(x) for x in cum_series]
        except Exception:
            c = [float(cum_series)]
    try:
        invest_val = float(invest_amount)
    except Exception:
        return math.inf
    for i in range(len(c)):
        val = c[i]
        if val >= invest_val:
            if i == 0:
                return 1.0
            prev = c[i-1]
            need = invest_val - prev
            delta = val - prev
            frac = 0.0 if delta == 0 else need / delta
            return i + frac
    return math.inf

# ---- laskenta ----
n_years = int(lifetime)
savings = np.array([(annual_saving * ((1 + saving_growth) ** t)) for t in range(n_years)], dtype=float)

loan_principal = float(invest) * float(financed_share)

payments = np.zeros(n_years, dtype=float)
interests = np.zeros(n_years, dtype=float)
principals = np.zeros(n_years, dtype=float)

if annuity:
    pay = annuity_payment(loan_principal, interest, int(loan_years))
    remaining = loan_principal
    for y in range(n_years):
        if y < loan_years:
            interest_part = remaining * interest
            principal_part = pay - interest_part
            payments[y] = pay
            interests[y] = max(0.0, interest_part)
            principals[y] = max(0.0, principal_part)
            remaining = max(0.0, remaining - principal_part)
else:
    for y in range(n_years):
        if y < loan_years:
            total, p_part, i_part = linear_payment(loan_principal, interest, int(loan_years), y + 1)
            payments[y] = total
            interests[y] = i_part
            principals[y] = p_part

om = np.full(n_years, float(om_fixed))
replacements = np.zeros(n_years, dtype=float)
if repl_interval and repl_cost and repl_interval > 0:
    for y in range(repl_interval, n_years, repl_interval):
        replacements[y] = float(repl_cost)

cashflow = savings - payments - om - replacements
cum_cash = np.cumsum(cashflow)

pb = payback_year(cum_cash, invest)

years = np.arange(1, n_years + 1)
discount_factors = 1.0 / np.power(1.0 + discount_rate, years)
npv = -float(invest) + float(np.sum(cashflow * discount_factors))

# ---- UI ----
col1, col2, col3 = st.columns(3)
col1.metric("NPV", f"{npv:,.0f} €")
col2.metric("Takaisinmaksuaika", "∞" if math.isinf(pb) else f"{pb:.1f} v")
col3.metric("1. vuoden kassavirta", f"{cashflow[0]:,.0f} €")

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("Vuosikassavirta (€)")
    fig = plt.figure()
    plt.plot(years, cashflow)
    plt.xlabel("Vuosi")
    plt.ylabel("€ / vuosi")
    st.pyplot(fig, clear_figure=True)

with right:
    st.subheader("Kumulatiivinen kassavirta (€)")
    fig2 = plt.figure()
    plt.plot(years, cum_cash)
    plt.axhline(float(invest), linestyle="--")
    plt.xlabel("Vuosi")
    plt.ylabel("€")
    st.pyplot(fig2, clear_figure=True)

st.divider()
st.subheader("Taulukko")
df = pd.DataFrame({
    "Vuosi": years,
    "Säästö (€)": np.round(savings, 2),
    "Lainan maksut (€)": np.round(payments, 2),
    "Korko (€)": np.round(interests, 2),
    "Lyhennys (€)": np.round(principals, 2),
    "Ylläpito (€)": np.round(om, 2),
    "Korvausinvestointi (€)": np.round(replacements, 2),
    "Nettokassavirta (€)": np.round(cashflow, 2),
    "Kumulatiivinen kassavirta (€)": np.round(cum_cash, 2),
    "Diskonttaustekijä": np.round(discount_factors, 5),
    "Diskontattu kassavirta (€)": np.round(cashflow * discount_factors, 2),
})
st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Lataa CSV", data=csv, file_name="npv_kassavirta.csv", mime="text/csv")
