import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# Configuración de página
st.set_page_config(page_title="Simulador Hipotecario Pro v2.7", layout="wide")

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
# 2. MOTOR ESTOCÁSTICO
# ==========================================
def simular_vasicek(r0, theta, kappa, sigma, anios, n_sims=100):
    dt = 1 
    simulaciones = []
    for _ in range(n_sims):
        camino = [r0]
        for t in range(anios - 1):
            dr = kappa * (theta - camino[-1]) * dt + sigma * np.random.normal()
            nuevo_r = camino[-1] + dr
            camino.append(max(-1.0, nuevo_r))
        simulaciones.append(camino)
    return np.array(simulaciones)

# ==========================================
# 3. INTERFAZ STREAMLIT
# ==========================================
st.title("🏦 Simulador Hipotecario: Monte Carlo v2.7")

with st.sidebar:
    st.header("⚙️ Configuración Préstamo")
    modo_h = st.selectbox("Modalidad", ["MIXTA", "VARIABLE", "FIJA"])
    tipo_reduc = st.radio("Reducir en...", ["PLAZO", "CUOTA"])
    capital = st.number_input("Capital Pendiente (€)", value=180000, step=1000)
    anios_p = st.number_input("Años Restantes", value=25, min_value=1)
    
    st.markdown("---")
    st.header("📈 Previsión Euríbor")
    modo_prev = st.radio("Método", ["Estocástico (Vasicek)", "Manual (Sliders)"])
    
    if modo_prev == "Estocástico (Vasicek)":
        n_sims = st.select_slider("Número de simulaciones", options=[10, 50, 100, 250, 500, 1000], value=100)
        theta = st.slider("Media L/P (θ)", 0.0, 5.0, 2.25, 0.05)
        sigma = st.slider("Volatilidad (σ)", 0.0, 2.0, 0.60, 0.05)
        kappa = st.slider("Vel. Reversión (κ)", 0.0, 1.0, 0.30, 0.05)
        r0 = st.number_input("Euríbor Inicial %", value=2.24)
    else:
        n_sims = 1 # En manual solo hay 1 escenario

    st.markdown("---")
    tipo_fijo = st.number_input("Tipo Fijo (%)", value=2.2, format="%.2f")
    anios_fijos = st.number_input("Años tramo fijo", value=7) if modo_h == "MIXTA" else 0
    diferencial = st.number_input("Diferencial Variable (%)", value=0.55, format="%.2f")

# --- Lógica de Euríbor ---
n_años_var = anios_p if modo_h == "VARIABLE" else max(0, anios_p - anios_fijos)

if modo_prev == "Manual (Sliders)":
    eur_list = []
    with st.expander("Configurar Euríbor Manual", expanded=True):
        cols = st.columns(4)
        for i in range(n_años_var):
            with cols[i % 4]:
                eur_list.append(st.slider(f"A{anios_p-n_años_var+i+1}", -1.0, 6.0, 2.25, key=f"e_{i}"))
    caminos_eur = [eur_list]
else:
    caminos_eur = simular_vasicek(r0, theta, kappa, sigma, n_años_var, n_sims=n_sims)

# --- Amortización ---
st.subheader("💰 Amortización Extra Anual")
amort_list = []
with st.expander("Configurar Pagos Extra", expanded=False):
    cols_a = st.columns(4)
    for i in range(anios_p):
        with cols_a[i % 4]:
            amort_list.append(st.slider(f"Año {i+1}", 0, 20000, 0, key=f"a_{i}", step=500))

# ==========================================
# 4. PROCESAMIENTO
# ==========================================
resultados_int = []
resultados_aho = []
df_central = None

# Progreso para simulaciones largas
if n_sims > 100:
    prog_bar = st.progress(0)
    
for i, camino in enumerate(caminos_eur):
    res = calcular_hipoteca(capital, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino, amort_list, tipo_reduc)
    res_b = calcular_hipoteca(capital, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino, [0]*anios_p, 'PLAZO')
    resultados_int.append(res['Intereses'].sum())
    resultados_aho.append(res_b['Intereses'].sum() - res['Intereses'].sum())
    
    # El escenario central es el primero (o podrías buscar el más cercano a la mediana)
    if df_central is None: df_central = res
    
    if n_sims > 100:
        prog_bar.progress((i + 1) / n_sims)

int_med, aho_med = np.median(resultados_int), np.median(resultados_aho)
p5, p95 = np.percentile(resultados_int, 5), np.percentile(resultados_int, 95)

# ==========================================
# 5. RESULTADOS
# ==========================================
c1, c2, c3 = st.columns(3)
c1.metric("Intereses (Mediana)", f"{int_med:,.0f} €")
c2.metric("Ahorro Amortización", f"{aho_med:,.0f} €")
c3.metric("Tiempo Ahorrado", f"{(anios_p*12 - len(df_central))//12}a {(anios_p*12 - len(df_central))%12}m")

if modo_prev == "Estocástico (Vasicek)":
    st.warning(f"⚠️ **Análisis de Riesgo ({n_sims} sims):** Hay un 5% de probabilidad de que los intereses superen los **{p95:,.0f} €**.")

st.markdown("---")
g1, g2 = st.columns(2)

with g1:
    st.write("**Previsión Euríbor (Intervalos de Confianza)**")
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(1, n_años_var + 1)
    if modo_prev == "Manual (Sliders)":
        ax.plot(x, caminos_eur[0], marker='o', color='#3498db')
    else:
        ax.plot(x, np.percentile(caminos_eur, 50, axis=0), color='#2980b9', label='Mediana')
        ax.fill_between(x, np.percentile(caminos_eur, 5, axis=0), np.percentile(caminos_eur, 95, axis=0), color='#2980b9', alpha=0.2, label='Rango P5-P95')
    ax.set_ylabel("Euríbor %")
    ax.legend()
    st.pyplot(fig)

with g2:
    st.write("**Cuota Mensual (Escenario Central)**")
    st.line_chart(df_central.set_index('Mes')['Cuota'])

# --- EXPORTACIÓN ---
st.markdown("---")
if st.checkbox("Ver cuadro de amortización detallado"):
    st.dataframe(df_central)
    
    # Botón para descargar CSV
    csv = df_central.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar cuadro en CSV",
        data=csv,
        file_name='cuadro_amortizacion.csv',
        mime='text/csv',
    )
