# -*- coding: utf-8 -*-
import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import traceback

st.set_page_config(page_title="Maalämpö vs. Kaukolämpö", page_icon="💶", layout="wide")
st.title("💶 Maalämpöinvestoinnin kannattavuus – yksi NPV, erillinen rahoitus")

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

def payback_year_from_cumulative(cumulative, threshold=0.0):
    for i, val in enumerate(cumulative):
        if val >= threshold:
            if i == 0:
                return 0.0
            prev = cumulative[i-1]
            need = threshold - prev
            delta = val - prev
            frac = 0.0 if delta == 0 else need / delta
            return i + frac
    return math.inf

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

    st.header("Rahoitus (vain kassavirtaan)")
    financed_share = st.slider("Lainan osuus investoinnista (%)", 0, 100, 100) / 100.0
    loan_years = st.number_input("Laina-aika (v)", min_value=1, value=20, step=1)
    interest = st.number_input("Korko (% p.a.)", min_value=0.0, value=4.0, step=0.1) / 100.0
    annuity = st.toggle("Annuiteettilaina", value=True)

    st.header("Diskonttaus (vain NPV:hen)")
    discount_rate = st.number_input("Diskonttokorko (% p.a.)", min_value=0.0, value=4.0, step=0.1) / 100.0

# --- Turvavyö: näytä virheet ruudulla ---
try:
    n_years = int(lifetime)
    years0 = np.arange(0, n_years + 1)
    years1 = np.arange(1, n_years + 1)

    # Kustannukset
    dh_costs = np.array([dh_price * ((1 + dh_infl) ** t) * dh_use for t in range(n_years)], dtype=float)
    ml_costs = np.array([elec_price * ((1 + elec_infl) ** t) * elec_use for t in range(n_years)], dtype=float)
    savings = dh_costs - ml_costs

    # 1) NPV ilman lainaa (oppikirja-NPV)
    discount_factors = 1.0 / np.power(1.0 + discount_rate, years1)
    npv_invest = -invest + np.sum(savings * discount_factors)

    cum_savings = np.zeros(n_years + 1)
    cum_savings[0] = -invest
    for i in range(1, n_years + 1):
        cum_savings[i] = cum_savings[i-1] + savings[i-1]
    pb_invest = payback_year_from_cumulative(cum_savings, threshold=0.0)

    # 2) Kassavirta lainalla (korot + lyhennys mukana) – EI NPV:tä
    loan_principal = invest * financed_share
    payments = np.zeros(n_years)
    if annuity:
        pay = annuity_payment(loan_principal, interest, int(loan_years))
        remaining = loan_principal
        for y in range(min(n_years, int(loan_years))):
            # korko + lyhennys
            interest_part = remaining * interest
            principal_part = pay - interest_part
            payments[y] = interest_part + principal_part
            remaining = max(0.0, remaining - principal_part)
    else:
        for y in range(min(n_years, int(loan_years))):
            payments[y] = linear_payment(loan_principal, interest, int(loan_years), y+1)

    cashflow_fin = savings - payments
    cum_cash_fin = np.zeros(n_years + 1)
    cum_cash_fin[0] = -invest
    for i in range(1, n_years + 1):
        cum_cash_fin[i] = cum_cash_fin[i-1] + cashflow_fin[i-1]
    pb_fin = payback_year_from_cumulative(cum_cash_fin, threshold=0.0)

    # --- Kortit ---
    c1, c2, c3 = st.columns(3)
    c1.metric("NPV (vain investointi)", f"{npv_invest:,.0f} €")
    c2.metric("Takaisinmaksuaika (ilman lainaa)", "∞" if math.isinf(pb_invest) else f"{pb_invest:.1f} v")
    c3.metric("Takaisinmaksuaika (lainalla)", "∞" if math.isinf(pb_fin) else f"{pb_fin:.1f} v")
    st.caption("NPV on laskettu ilman lainanhoitoa. Lainan korot ja lyhennykset huomioidaan vain rahoituksen kassavirrassa ja takaisinmaksuajassa.")

    st.divider()

    # Kuvaaja A
    st.subheader("Kumulatiivinen säästö (ilman lainaa) – alkaa investoinnista")
    figA = plt.figure()
    plt.plot(years0, cum_savings, marker="o")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Vuosi")
    plt.ylabel("€")
    st.pyplot(figA, clear_figure=True)

    # Kuvaaja B
    st.subheader("Kumulatiivinen kassavirta (lainalla) – korot mukana")
    figB = plt.figure()
    plt.plot(years0, cum_cash_fin, marker="o")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Vuosi")
    plt.ylabel("€")
    st.pyplot(figB, clear_figure=True)

except Exception as e:
    st.error("Sovellus kaatui – virhe tulostettu alle.")
    st.exception(e)
    st.code(traceback.format_exc())
