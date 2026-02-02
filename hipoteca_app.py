import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Configuración de página
st.set_page_config(page_title="Simulador Hipotecario & Patrimonial v2.9", layout="wide")

# ==========================================
# 1. NÚCLEO MATEMÁTICO (Auditado)
# ==========================================
def calcular_hipoteca(capital, anios, diferencial, tipo_fijo, anios_fijos, modo, euribor_puntos, amortizaciones, tipo_reduc):
    n_meses_total = anios * 12
    saldo_real = round(float(capital), 2)
    saldo_teorico = round(float(capital), 2)
    data = []
    mes_global = 1
    
    puntos_eur = list(euribor_puntos) + [euribor_puntos[-1]] * (max(0, anios - len(euribor_puntos)))
    puntos_amort = list(amortizaciones) + [0] * (max(0, anios - len(amortizaciones)))

    idx_var = 0 
    for anio in range(anios):
        if saldo_real <= 0: break
        tasa_anual = tipo_fijo if (modo == 'FIJA' or (modo == 'MIXTA' and anio < anios_fijos)) else (puntos_eur[idx_var if modo == 'MIXTA' else anio] + diferencial)
        if modo == 'MIXTA' and anio >= anios_fijos: idx_var += 1
        
        tasa_mensual = (max(0, tasa_anual) / 100) / 12
        meses_restantes = n_meses_total - (mes_global - 1)
        base_calc = saldo_teorico if tipo_reduc == 'PLAZO' else saldo_real
        if base_calc < saldo_real: base_calc = saldo_real

        cuota = round(base_calc * (tasa_mensual * (1 + tasa_mensual)**meses_restantes) / ((1 + tasa_mensual)**meses_restantes - 1), 2) if tasa_mensual > 0 else round(base_calc / meses_restantes, 2)

        for m in range(12):
            if saldo_real <= 0.009: break
            interes_m = round(saldo_real * tasa_mensual, 2)
            capital_m = round(cuota - interes_m, 2)
            if capital_m > saldo_real:
                capital_m = saldo_real
                cuota = round(capital_m + interes_m, 2)

            saldo_real = round(saldo_real - capital_m, 2)
            saldo_teorico = round(saldo_teorico - (cuota - round(saldo_teorico * tasa_mensual, 2)), 2)

            data.append({'Mes': mes_global, 'Año': anio + 1, 'Tasa': tasa_anual, 'Cuota': cuota, 'Intereses': interes_m, 'Capital': capital_m, 'Saldo': saldo_real, 'Amort_Extra': 0})
            
            if m == 11 and saldo_real > 0 and puntos_amort[anio] > 0:
                ejec = round(min(puntos_amort[anio], saldo_real), 2)
                saldo_real = round(saldo_real - ejec, 2)
                if tipo_reduc == 'CUOTA': saldo_teorico = saldo_real
                data[-1]['Amort_Extra'] = ejec
                data[-1]['Capital'] = round(data[-1]['Capital'] + ejec, 2)
            mes_global += 1

    return pd.DataFrame(data)

def simular_vasicek(r0, theta, kappa, sigma, anios, n_sims=100):
    dt = 1 
    sims = []
    for _ in range(n_sims):
        camino = [r0]
        for t in range(anios - 1):
            camino.append(max(-1.0, camino[-1] + kappa * (theta - camino[-1]) * dt + sigma * np.random.normal()))
        sims.append(camino)
    return np.array(sims)

# ==========================================
# 2. INTERFAZ Y CONFIGURACIÓN
# ==========================================
st.title("🏦 Simulador Patrimonial & Hipotecario v2.9")

with st.sidebar:
    st.header("👤 Perfil Financiero")
    ingresos = st.number_input("Ingresos Netos Mensuales (€)", value=3000, step=100)
    ahorro_inicial = st.number_input("Ahorro Líquido Inicial (€)", value=20000, step=1000)
    precio_vivienda = st.number_input("Precio Compra Vivienda (€)", value=220000, step=5000)
    
    st.header("⚙️ Préstamo")
    modo_h = st.selectbox("Modalidad", ["MIXTA", "VARIABLE", "FIJA"])
    tipo_reduc = st.radio("Reducción por Amortización", ["PLAZO", "CUOTA"])
    capital_init = st.number_input("Capital Pendiente (€)", value=180000, step=1000)
    anios_p = st.number_input("Años Restantes", value=25, min_value=1)
    
    st.header("🛡️ Gastos y Seguros")
    s_hogar = st.number_input("Seguro Hogar (Anual €)", value=350)
    s_vida = st.number_input("Seguro Vida (Anual €)", value=400)
    gastos_fijos = st.number_input("Comunidad/IBI/Otros (Mensual €)", value=120)

    st.header("📈 Euríbor")
    modo_prev = st.radio("Método", ["Estocástico (Vasicek)", "Manual (Sliders)"])
    n_sims = st.select_slider("Simulaciones", options=[10, 50, 100, 250], value=50) if modo_prev == "Estocástico (Vasicek)" else 1
    
    if modo_prev == "Estocástico (Vasicek)":
        theta, sigma, kappa = 2.25, 0.60, 0.30
        r0 = 2.24
    
    st.markdown("---")
    tipo_fijo = st.number_input("Tipo Fijo (%)", value=2.2)
    anios_fijos = st.number_input("Años fijo", value=7) if modo_h == "MIXTA" else 0
    diferencial = st.number_input("Diferencial %", value=0.55)

# Procesamiento de caminos
n_años_var = anios_p if modo_h == "VARIABLE" else max(0, anios_p - anios_fijos)
caminos_eur = simular_vasicek(r0, theta, kappa, sigma, n_años_var, n_sims) if modo_prev == "Estocástico (Vasicek)" else [[2.25]*n_años_var]
amort_list = [0]*anios_p # Simplificado para el ejemplo o usar expander anterior

# ==========================================
# 3. CÁLCULOS PATRIMONIALES
# ==========================================
all_results = []
for camino in caminos_eur:
    df = calcular_hipoteca(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino, amort_list, tipo_reduc)
    
    # Añadir lógica de gastos
    gasto_mensual_extra = (s_hogar + s_vida) / 12 + gastos_fijos
    df['Gasto_Total'] = df['Cuota'] + gasto_mensual_extra
    df['Ahorro_Mensual'] = ingresos - df['Gasto_Total']
    
    # Evolución Capital
    df['Ahorro_Acumulado'] = ahorro_inicial + df['Ahorro_Mensual'].cumsum() - df['Amort_Extra'].cumsum()
    df['Equity'] = precio_vivienda - df['Saldo']
    df['Patrimonio_Neto'] = df['Ahorro_Acumulado'] + df['Equity']
    
    all_results.append(df)

df_c = all_results[0] # Escenario central

# ==========================================
# 4. DASHBOARD
# ==========================================
m1, m2, m3, m4 = st.columns(4)
m1.metric("Patrimonio Final Est.", f"{df_c['Patrimonio_Neto'].iloc[-1]:,.0f} €")
m2.metric("Esfuerzo Medio", f"{(df_c['Gasto_Total'].mean()/ingresos*100):.1f}%")
m3.metric("Intereses Totales", f"{df_c['Intereses'].sum():,.0f} €")
m4.metric("Capital Líquido Final", f"{df_c['Ahorro_Acumulado'].iloc[-1]:,.0f} €")

st.markdown("---")
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.subheader("📈 Evolución del Patrimonio Neto")
    fig_pn = go.Figure()
    fig_pn.add_trace(go.Scatter(x=df_c['Mes'], y=df_c['Patrimonio_Neto'], name="Patrimonio Neto Total", line=dict(color='#8e44ad', width=4)))
    fig_pn.add_trace(go.Scatter(x=df_c['Mes'], y=df_c['Ahorro_Acumulado'], name="Ahorro Líquido", fill='tozeroy', line=dict(color='#27ae60')))
    fig_pn.add_trace(go.Scatter(x=df_c['Mes'], y=df_c['Equity'], name="Valor Propio Casa (Equity)", fill='tonexty', line=dict(color='#2980b9')))
    fig_pn.update_layout(hovermode="x unified", height=450)
    st.plotly_chart(fig_pn, use_container_width=True)

with col_g2:
    st.subheader("💸 Desglose de Gastos Mensuales")
    fig_g = go.Figure()
    fig_g.add_trace(go.Bar(x=df_c['Mes'], y=df_c['Cuota'], name="Cuota Hipoteca", marker_color='#2980b9'))
    fig_g.add_trace(go.Bar(x=df_c['Mes'], y=[(s_hogar+s_vida)/12]*len(df_c), name="Seguros", marker_color='#e67e22'))
    fig_g.add_trace(go.Bar(x=df_c['Mes'], y=[gastos_fijos]*len(df_c), name="Otros Gastos", marker_color='#95a5a6'))
    fig_g.update_layout(barmode='stack', hovermode="x unified", height=450)
    st.plotly_chart(fig_g, use_container_width=True)

if st.checkbox("Ver informe detallado"):
    st.dataframe(df_c[['Mes', 'Cuota', 'Saldo', 'Ahorro_Acumulado', 'Patrimonio_Neto']].style.format("{:.2f}"))
