import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuración de página
st.set_page_config(page_title="Simulador Hipotecario Pro (Stochastic v2.5)", layout="wide")

# ==========================================
# 1. NÚCLEO MATEMÁTICO (Auditado)
# ==========================================
def calcular_hipoteca(capital, anios, diferencial, tipo_fijo, anios_fijos, modo, euribor_puntos, amortizaciones, tipo_reduc):
    n_meses_total = anios * 12
    saldo_real = round(float(capital), 2)
    saldo_teorico = round(float(capital), 2)
    data = []
    mes_global = 1
    
    # Asegurar longitud de listas (relleno por el final)
    puntos_eur = list(euribor_puntos) + [euribor_puntos[-1]] * (max(0, anios - len(euribor_puntos)))
    puntos_amort = list(amortizaciones) + [0] * (max(0, anios - len(amortizaciones)))

    idx_var = 0 
    for anio in range(anios):
        if saldo_real <= 0: break

        if modo == 'FIJA':
            tasa_anual = tipo_fijo
        elif modo == 'VARIABLE':
            tasa_anual = puntos_eur[anio] + diferencial
        else: # MIXTA
            if anio < anios_fijos:
                tasa_anual = tipo_fijo
            else:
                val_eur = puntos_eur[idx_var] if idx_var < len(puntos_eur) else puntos_eur[-1]
                tasa_anual = val_eur + diferencial
                idx_var += 1

        tasa_mensual = (max(0, tasa_anual) / 100) / 12
        meses_restantes = n_meses_total - (mes_global - 1)
        base_calc = saldo_teorico if tipo_reduc == 'PLAZO' else saldo_real
        if base_calc < saldo_real: base_calc = saldo_real

        if base_calc <= 0.01: cuota = 0
        else:
            if tasa_mensual > 0:
                cuota = base_calc * (tasa_mensual * (1 + tasa_mensual)**meses_restantes) / ((1 + tasa_mensual)**meses_restantes - 1)
            else: cuota = base_calc / meses_restantes
        
        cuota = round(cuota, 2)

        for m in range(12):
            if saldo_real <= 0.009: break
            interes_m = round(saldo_real * tasa_mensual, 2)
            capital_m = round(cuota - interes_m, 2)
            if capital_m > saldo_real:
                capital_m = saldo_real
                cuota = round(capital_m + interes_m, 2)

            saldo_real = round(saldo_real - capital_m, 2)
            int_teorico = round(saldo_teorico * tasa_mensual, 2)
            amort_teorica = round(cuota - int_teorico, 2)
            saldo_teorico = round(saldo_teorico - amort_teorica, 2)

            data.append({'Mes': mes_global, 'Año': anio + 1, 'Tasa': tasa_anual, 'Cuota': cuota, 'Intereses': interes_m, 'Saldo': saldo_real, 'Amort_Extra': 0})
            
            if m == 11 and saldo_real > 0 and puntos_amort[anio] > 0:
                ejec = round(min(puntos_amort[anio], saldo_real), 2)
                saldo_real = round(saldo_real - ejec, 2)
                if tipo_reduc == 'CUOTA': saldo_teorico = saldo_real
                data[-1]['Amort_Extra'] = ejec
            mes_global += 1

    return pd.DataFrame(data)

# ==========================================
# 2. MOTOR ESTOCÁSTICO (Modelo Vasicek)
# ==========================================
def simular_vasicek(r0, theta, kappa, sigma, anios, n_sims=100):
    """Genera n_sims caminos de tipos de interés."""
    dt = 1 # Paso anual
    simulaciones = []
    for _ in range(n_sims):
        camino = [r0]
        for t in range(anios - 1):
            dr = kappa * (theta - camino[-1]) * dt + sigma * np.random.normal()
            nuevo_r = camino[-1] + dr
            camino.append(max(-1.0, nuevo_r)) # Límite suelo -1%
        simulaciones.append(camino)
    return np.array(simulaciones)

# ==========================================
# 3. INTERFAZ STREAMLIT
# ==========================================
st.title("🏦 Simulador Hipotecario Profesional v2.5")

with st.sidebar:
    st.header("⚙️ Configuración Préstamo")
    modo_h = st.selectbox("Modalidad Préstamo", ["MIXTA", "VARIABLE", "FIJA"])
    tipo_reduc = st.radio("Estrategia Amortización", ["PLAZO", "CUOTA"])
    capital = st.number_input("Capital Pendiente (€)", value=180000, step=1000)
    anios_p = st.number_input("Plazo Restante (Años)", value=25, min_value=1)
    
    st.markdown("---")
    st.header("📈 Modo Previsión Euríbor")
    modo_prevision = st.radio("Método de cálculo", ["Manual (Sliders)", "Estocástico (Monte Carlo)"])
    
    if modo_prevision == "Estocástico (Monte Carlo)":
        st.info("Modelo Vasicek: Reversión a la media.")
        theta = st.slider("Media L/P (θ) %", 0.0, 5.0, 2.5, 0.1)
        sigma = st.slider("Volatilidad (σ)", 0.0, 2.0, 0.8, 0.1)
        kappa = st.slider("Vel. Reversión (κ)", 0.0, 1.0, 0.3, 0.1)
        r0 = st.number_input("Euríbor Inicial %", value=2.5)
    
    st.markdown("---")
    tipo_fijo = st.number_input("Tipo Fijo (%)", value=2.2, format="%.2f")
    anios_fijos = st.number_input("Años de tramo fijo", value=7) if modo_h == "MIXTA" else 0
    diferencial = st.number_input("Diferencial Variable (%)", value=0.55, format="%.2f")

# --- Generación de Datos de Euríbor ---
n_años_var = anios_p if modo_h == "VARIABLE" else max(0, anios_p - anios_fijos)

if modo_prevision == "Manual (Sliders)":
    st.subheader("📉 Previsión Euríbor Manual")
    eur_list = []
    with st.expander("Configurar Euríbor año a año", expanded=True):
        cols = st.columns(4)
        for i in range(n_años_var):
            with cols[i % 4]:
                val = st.slider(f"A{anios_p-n_años_var+i+1}", -1.0, 6.0, 2.5, key=f"eur_{i}", step=0.1)
                eur_list.append(val)
    # Lista de una sola simulación
    caminos_eur = [eur_list]
else:
    st.subheader("🎲 Simulación de Monte Carlo (100 escenarios)")
    caminos_eur = simular_vasicek(r0, theta, kappa, sigma, n_años_var, n_sims=100)

# --- Sliders Amortización ---
st.subheader("💰 Amortización Extra Anual")
amort_list = []
with st.expander("Configurar Pagos Extra", expanded=False):
    cols_a = st.columns(4)
    for i in range(anios_p):
        with cols_a[i % 4]:
            val_a = st.slider(f"Año {i+1}", 0, 20000, 0, key=f"am_{i}", step=500)
            amort_list.append(val_a)

# ==========================================
# 4. PROCESAMIENTO MULTI-ESCENARIO
# ==========================================
resultados_intereses = []
resultados_ahorro = []
df_final_median = None

# Corremos la simulación para cada camino de Euríbor generado
for camino in caminos_eur:
    res = calcular_hipoteca(capital, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino, amort_list, tipo_reduc)
    # Escenario base para ROI
    res_b = calcular_hipoteca(capital, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino, [0]*anios_p, 'PLAZO')
    
    resultados_intereses.append(res['Intereses'].sum())
    resultados_ahorro.append(res_b['Intereses'].sum() - res['Intereses'].sum())
    
    # Guardamos el primer dataframe (o el central) para visualización
    if df_final_median is None: df_final_median = res

# Estadísticas finales
interes_medio = np.median(resultados_intereses)
ahorro_medio = np.median(resultados_ahorro)
p5_int, p95_int = np.percentile(resultados_intereses, 5), np.percentile(resultados_intereses, 95)

# ==========================================
# 5. RENDERIZADO RESULTADOS
# ==========================================
m1, m2, m3 = st.columns(3)
if modo_prevision == "Manual (Sliders)":
    m1.metric("Intereses Totales", f"{interes_medio:,.0f} €")
    m2.metric("Ahorro por Amortización", f"{ahorro_medio:,.0f} €")
else:
    m1.metric("Intereses (Mediana)", f"{interes_medio:,.0f} €", help=f"Rango Probable (P5-P95): {p5_int:,.0f}€ - {p95_int:,.0f}€")
    m2.metric("Ahorro Medio", f"{ahorro_medio:,.0f} €")

m3.metric("Tiempo Ahorrado", f"{(anios_p*12 - len(df_final_median))//12}a {(anios_p*12 - len(df_final_median))%12}m")

st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    st.write("**Euríbor: Escenario Central vs Incertidumbre**")
    fig_eur, ax_eur = plt.subplots(figsize=(10, 4))
    x_axis = np.arange(1, n_años_var + 1)
    if modo_prevision == "Manual (Sliders)":
        ax_eur.plot(x_axis, caminos_eur[0], marker='o', color='#2980b9')
    else:
        p05 = np.percentile(caminos_eur, 5, axis=0)
        p50 = np.percentile(caminos_eur, 50, axis=0)
        p95 = np.percentile(caminos_eur, 95, axis=0)
        ax_eur.plot(x_axis, p50, color='#2980b9', label='Mediana')
        ax_eur.fill_between(x_axis, p05, p95, color='#2980b9', alpha=0.2, label='Rango Confianza 90%')
    ax_eur.set_ylabel("Euríbor %")
    ax_eur.grid(True, alpha=0.3)
    ax_eur.legend()
    st.pyplot(fig_eur)

with c2:
    st.write("**Cuota Mensual Estimada**")
    st.line_chart(df_final_median.set_index('Mes')['Cuota'])

if st.checkbox("Ver tabla detallada (Escenario Central)"):
    st.dataframe(df_final_median)
