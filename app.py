# -*- coding: utf-8 -*-
import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Maalämpö vs. Kaukolämpö", page_icon="💶", layout="wide")
st.title("💶 Maalämpöinvestoinnin kannattavuus")

# --- Syötteet ---
with st.sidebar:
    st.header("Investointi ja elinkaari")
    invest = st.number_input("Maalämmön investointi (€)", min_value=0, value=600000, step=10000, format="%d")
    lifetime = st.number_input("Elinkaari (vuotta)", min_value=1, value=25, step=1)

    st.header("Kaukolämpö")
    dh_price = st.number_input("Kaukolämmön hinta (€/MWh)", min_value=0.0, value=100.0, step=1.0)
    dh_use = st.number_input("Kaukolämmön kulutus (MWh/v)", min_value=0.0, value=500.0, step=10.0)
    dh_infl = st.number_input("Kaukolämmön inflaatio (%/v)", value=2.0, step=0.1) / 100.0

    st.header("Maalämpö")
    elec_use = st.number_input("Maalämmön sähkönkulutus (MWh/v)", min_value=0.0, value=150.0, step=10.0)
    elec_price = st.number_input("Sähkön hinta (€/MWh)", min_value=0.0, value=60.0, step=1.0)
    elec_infl = st.number_input("Sähkön inflaatio (%/v)", value=2.0, step=0.1) / 100.0

    st.header("Rahoitus")
    financed_share = st.slider("Lainan osuus investoinnista (%)", 0, 100, 100) / 100.0
    loan_years = st.number_input("Laina-aika (v)", min_value=1, value=20, step=1)
    interest = st.number_input("Korko (% p.a.)", min_value=0.0, value=4.0, step=0.1) / 100.0
    annuity = st.toggle("Annuiteettilaina", value=True)

    st.header("NPV")
    discount_rate = st.number_input("Diskonttokorko (% p.a.)", min_value=0.0, value=4.0, step=0.1) / 100.0

# --- Apufunktiot ---
def annuity_payment(principal, r, n):
    if principal <= 0 or n <= 0:
        return 0.0
    if r == 0:
        return principal / n
    return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

def linear_payment(principal, r, n, year):
    if principal <= 0 or n <= 0:
        return 0.0
    principal_payment = principal / n
    remaining = principal - (year - 1) * principal_payment
    interest_payment = remaining * r
    return principal_payment + interest_payment

def payback_year(cumulative):
    for i, val in enumerate(cumulative):
        if val >= 0:
            if i == 0:
                return 0.0
            prev = cumulative[i-1]
            need = 0 - prev
            delta = val - prev
            frac = 0.0 if delta == 0 else need / delta
            return (i) + frac
    return math.inf

# --- Laskenta ---
n_years = int(lifetime)
years = np.arange(0, n_years + 1)

# Kaukolämmön kustannukset
dh_costs = np.array([dh_price * ((1 + dh_infl) ** t) * dh_use for t in range(n_years)], dtype=float)

# Maalämmön kustannukset
ml_costs = np.array([elec_price * ((1 + elec_infl) ** t) * elec_use for t in range(n_years)], dtype=float)

# Säästöt (lämmityskustannusten ero)
savings = dh_costs - ml_costs

# Lainan maksut
loan_principal = invest * financed_share
payments = np.zeros(n_years)
if annuity:
    pay = annuity_payment(loan_principal, interest, loan_years)
    remaining = loan_principal
    for y in range(n_years):
        if y < loan_years:
            payments[y] = pay
            remaining = remaining * (1 + interest) - pay
else:
    for y in range(n_years):
        if y < loan_years:
            payments[y] = linear_payment(loan_principal, interest, loan_years, y+1)

# --- NPV ilman lainaa (puhdas investointi) ---
discount_factors = 1.0 / np.power(1.0 + discount_rate, np.arange(1, n_years + 1))
npv_invest = -invest + np.sum(savings * discount_factors)

# --- Kassavirta lainalla (osakkaan näkökulma) ---
cashflow = savings - payments
cum_cash = np.zeros(n_years + 1)
cum_cash[0] = -invest
for i in range(1, n_years + 1):
    cum_cash[i] = cum_cash[i-1] + cashflow[i-1]

pb = payback_year(cum_cash)
discounted_cash = cashflow * discount_factors
npv_finance = -invest + np.sum(discounted_cash)

# --- Tulokset ---
col1, col2, col3 = st.columns(3)
col1.metric("NPV (ilman lainaa)", f"{npv_invest:,.0f} €")
col2.metric("NPV (lainalla)", f"{npv_finance:,.0f} €")
col3.metric("Takaisinmaksuaika", "∞" if math.isinf(pb) else f"{pb:.1f} v")

st.divider()

# --- Kuvaaja 1: Kumulatiivinen kassavirta (lainalla) ---
st.subheader("Kumulatiivinen kassavirta (lainalla)")
fig1 = plt.figure()
plt.plot(years, cum_cash, marker="o")
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Vuosi")
plt.ylabel("€")
st.pyplot(fig1, clear_figure=True)

# --- Kuvaaja 2: Diskontatut säästöt (ilman lainaa) ---
st.subheader("Diskontatut säästöt ja NPV (ilman lainaa)")
fig2 = plt.figure()
plt.bar(np.arange(1, n_years+1), savings * discount_factors, label="Diskontatut säästöt")
plt.plot(np.arange(1, n_years+1), np.cumsum(savings * discount_factors) - invest,
         color="red", marker="o", label="Kumulatiivinen (NPV)")
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Vuosi")
plt.ylabel("€ (nykyarvo)")
plt.legend()
st.pyplot(fig2, clear_figure=True)
