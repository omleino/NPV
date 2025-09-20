# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NPV – Ultra-minimi", page_icon="💶", layout="centered")

st.title("💶 NPV – Ultra-minimi")

with st.sidebar:
    invest = st.number_input("Investointi (€)", min_value=0.0, value=600000.0, step=10000.0, format="%.0f")
    lifetime = st.number_input("Elinkaari (v)", min_value=1, value=25, step=1)
    annual_saving = st.number_input("Vuosisäästö 1. vuotena (€ / v)", min_value=0.0, value=50000.0, step=1000.0, format="%.0f")
    saving_growth = st.number_input("Säästön kasvu (% / v)", min_value=-50.0, max_value=50.0, value=0.0, step=0.1) / 100.0
    financed_share = st.slider("Lainan osuus (%)", 0, 100, 100) / 100.0
    loan_years = st.number_input("Laina-aika (v)", min_value=1, value=20, step=1)
    interest = st.number_input("Korko (% p.a.)", min_value=0.0, value=4.0, step=0.1) / 100.0
    annuity = st.toggle("Annuiteetti", value=True)
    discount_rate = st.number_input("Diskonttokorko (% p.a.)", min_value=0.0, value=4.0, step=0.1) / 100.0
    om_fixed = st.number_input("Ylläpito (€ / v)", min_value=0.0, value=0.0, step=500.0, format="%.0f")

def annuity_payment(principal, r, n):
    if principal <= 0 or n <= 0: return 0.0
    if r == 0: return principal / n
    return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

def linear_payment(principal, r, n, year):
    if principal <= 0 or n <= 0: return 0.0, 0.0, 0.0
    principal_payment = principal / n
    remaining = principal - (year - 1) * principal_payment
    interest_payment = remaining * r
    total = principal_payment + interest_payment
    return total, principal_payment, interest_payment

def payback_year(cum_series, invest_amount):
    c = np.asarray(cum_series, dtype=float).ravel()
    inv = float(invest_amount)
    for i in range(len(c)):
        if c[i] >= inv:
            if i == 0: return 1.0
            prev = c[i-1]; need = inv - prev; delta = c[i] - prev
            frac = 0.0 if delta == 0 else need / delta
            return i + frac
    return math.inf

n_years = int(lifetime)
years = np.arange(1, n_years + 1)
savings = np.array([(annual_saving * ((1 + saving_growth) ** t)) for t in range(n_years)], dtype=float)
loan_principal = float(invest) * float(financed_share)

payments = np.zeros(n_years, dtype=float)
interests = np.zeros(n_years, dtype=float)
principals = np.zeros(n_years, dtype=float)

if annuity:
    pay = annuity_payment(loan_principal, interest, int(loan_years))
    remaining = loan_principal
    for y in range(min(n_years, int(loan_years))):
        interest_part = remaining * interest
        principal_part = pay - interest_part
        payments[y] = pay
        interests[y] = max(0.0, interest_part)
        principals[y] = max(0.0, principal_part)
        remaining = max(0.0, remaining - principal_part)
else:
    for y in range(min(n_years, int(loan_years))):
        total, p_part, i_part = linear_payment(loan_principal, interest, int(loan_years), y + 1)
        payments[y] = total
        interests[y] = i_part
        principals[y] = p_part

om = np.full(n_years, float(om_fixed))
cashflow = savings - payments - om
cum_cash = np.cumsum(cashflow)

pb = payback_year(cum_cash, invest)
discount_factors = 1.0 / np.power(1.0 + discount_rate, years)
npv = -float(invest) + float(np.sum(cashflow * discount_factors))

st.metric("NPV", f"{npv:,.0f} €")
st.metric("Takaisinmaksuaika (ei disk.)", "∞" if math.isinf(pb) else f"{pb:.1f} v")
st.metric("1. vuoden nettokassavirta", f"{cashflow[0]:,.0f} €")

df = pd.DataFrame({
    "Vuosi": years,
    "Säästö (€)": np.round(savings, 2),
    "Lainan maksut (€)": np.round(payments, 2),
    "Korko (€)": np.round(interests, 2),
    "Lyhennys (€)": np.round(principals, 2),
    "Ylläpito (€)": np.round(om, 2),
    "Nettokassavirta (€)": np.round(cashflow, 2),
    "Kumulatiivinen kassavirta (€)": np.round(cum_cash, 2),
    "Diskonttaustekijä": np.round(discount_factors, 5),
    "Diskontattu kassavirta (€)": np.round(cashflow * discount_factors, 2),
})
st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Lataa CSV", data=csv, file_name="npv_ultra.csv", mime="text/csv")

st.caption("Ultra-minimi: ei matplotlibia, ei PDF:ää. Jos tämä toimii Cloudissa, vika on grafiikka-/kirjastoriippuvuuksissa.")
