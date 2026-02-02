import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# CONFIGURACIÓN
# ==========================================
st.set_page_config(page_title="Simulador Hipotecario Pro 3.5", page_icon="🏦", layout="wide")

# ==========================================
# 1. MOTOR MATEMÁTICO (CORE)
# ==========================================
def calcular_hipoteca_core(capital, anios, diferencial, tipo_fijo, anios_fijos, modo, euribor_puntos, amortizaciones, tipo_reduc):
    n_meses_total = anios * 12
    saldo_real = round(float(capital), 2)
    saldo_teorico = round(float(capital), 2)
    data = []
    mes_global = 1
    
    # Relleno seguro de listas
    puntos_eur = list(euribor_puntos) + [euribor_puntos[-1]] * (max(0, anios - len(euribor_puntos)))
    puntos_amort = list(amortizaciones) + [0] * (max(0, anios - len(amortizaciones)))

    idx_var = 0 
    for anio in range(anios):
        if saldo_real <= 0: break 

        # LÓGICA DE TIPO DE INTERÉS SEGÚN MODO
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
        
        # Base cálculo cuota
        base_calc = saldo_teorico if tipo_reduc == 'PLAZO' else saldo_real
        if base_calc < saldo_real: base_calc = saldo_real

        if base_calc <= 0.01: cuota = 0
        else:
            if tasa_mensual > 0:
                cuota = base_calc * (tasa_mensual * (1 + tasa_mensual)**meses_restantes) / ((1 + tasa_mensual)**meses_restantes - 1)
            else: cuota = base_calc / meses_restantes
        
        cuota = round(cuota, 2)

        for m in range(12):
            if saldo_real <= 0.009: 
                saldo_real = 0
                break

            interes_m = round(saldo_real * tasa_mensual, 2)
            capital_m = round(cuota - interes_m, 2)
            
            if capital_m > saldo_real:
                capital_m = saldo_real
                cuota = round(capital_m + interes_m, 2)

            saldo_real = round(saldo_real - capital_m, 2)
            
            # Ajuste saldo teórico
            int_teorico = round(saldo_teorico * tasa_mensual, 2)
            amort_teorica = round(cuota - int_teorico, 2)
            saldo_teorico = round(saldo_teorico - amort_teorica, 2)
            if saldo_teorico < 0: saldo_teorico = 0

            data.append({
                'Mes': mes_global, 'Año': anio + 1, 'Tasa': tasa_anual, 
                'Cuota': cuota, 'Intereses': interes_m, 'Capital': capital_m, 
                'Saldo': saldo_real, 'Amort_Extra': 0
            })
            
            # Amortización Extra
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
            dr = kappa * (theta - camino[-1]) * dt + sigma * np.random.normal()
            camino.append(max(-1.0, camino[-1] + dr))
        sims.append(camino)
    return np.array(sims)

# ==========================================
# 2. INTERFAZ DINÁMICA (SIDEBAR)
# ==========================================
st.title("🏦 Simulador Hipotecario Pro 3.5")
st.markdown("---")

with st.sidebar:
    st.header("👤 Perfil Financiero")
    ingresos = st.number_input("Ingresos Mensuales (€)", value=3000, step=100)
    ahorro_inicial = st.number_input("Ahorro Inicial (€)", value=20000, step=1000)
    precio_vivienda = st.number_input("Valor Vivienda (€)", value=220000, step=5000)
    
    st.markdown("---")
    st.header("⚙️ Estructura Préstamo")
    
    # DATOS COMUNES
    modo_h = st.selectbox("Modalidad", ["MIXTA", "VARIABLE", "FIJA"])
    capital_init = st.number_input("Capital Pendiente (€)", value=180000, step=1000)
    anios_p = st.number_input("Años Restantes", value=25, min_value=1)
    tipo_reduc = st.radio("Estrategia Amortización", ["PLAZO", "CUOTA"])
    
    st.markdown("---")
    st.header("🏦 Condiciones Banco")
    
    # VARIABLES DINÁMICAS (Lógica de visibilidad)
    tipo_fijo = 0.0
    diferencial = 0.0
    anios_fijos = 0
    
    if modo_h == "FIJA":
        tipo_fijo = st.number_input("Tipo Fijo (%)", value=2.50, step=0.05)
        # Euribor y Diferencial no aplican
        
    elif modo_h == "VARIABLE":
        diferencial = st.number_input("Diferencial (%)", value=0.55, step=0.05)
        # Tipo Fijo no aplica
        
    elif modo_h == "MIXTA":
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            tipo_fijo = st.number_input("Tipo Fijo (%)", value=2.25, step=0.05)
        with col_m2:
            anios_fijos = st.number_input("Años Fijos", value=5, min_value=1, max_value=anios_p-1)
        diferencial = st.number_input("Diferencial (%)", value=0.55, step=0.05)

    st.markdown("---")
    st.header("🛡️ Gastos")
    s_hogar = st.number_input("Seguro Hogar/año (€)", value=300)
    s_vida = st.number_input("Seguro Vida/año (€)", value=400)
    gastos_fijos = st.number_input("Comunidad/mes (€)", value=100)


# ==========================================
# 3. CONFIGURACIÓN EURÍBOR (SOLO SI APLICA)
# ==========================================
caminos_eur = []
modo_prev = "N/A"
n_sims = 1

# Solo mostramos configuración de Euribor si NO es Fija
if modo_h != "FIJA":
    # Sidebar extra para Euribor solo si es necesario
    with st.sidebar:
        st.markdown("---")
        st.header("📈 Config. Euríbor")
        modo_prev = st.radio("Método", ["Estocástico (Monte Carlo)", "Manual (Sliders)"])
        
        if modo_prev == "Estocástico (Monte Carlo)":
            n_sims = st.select_slider("Simulaciones", [10, 50, 100, 250, 500], value=100)
            theta = st.slider("Media (θ)", 0.0, 5.0, 2.25)
            sigma = st.slider("Volatilidad (σ)", 0.0, 2.0, 0.60)
            kappa = st.slider("Inercia (κ)", 0.0, 1.0, 0.30)
            r0 = st.number_input("Euríbor Hoy", value=2.24)
        else:
            n_sims = 1

    # Generación de Caminos en Main Page
    n_años_var = anios_p if modo_h == "VARIABLE" else max(0, anios_p - anios_fijos)
    
    if n_años_var > 0:
        if modo_prev == "Manual (Sliders)":
            with st.expander("🛠️ Ajustar Euríbor Manualmente", expanded=True):
                eur_list = [st.slider(f"A{i+1}", -1.0, 7.0, 2.25, key=f"e{i}") for i in range(n_años_var)]
            caminos_eur = [eur_list]
        else:
            # Info Vasicek
            caminos_eur = simular_vasicek(r0, theta, kappa, sigma, n_años_var, n_sims)
else:
    # Si es FIJA, creamos un camino dummy de ceros para que no falle el código, 
    # aunque la función core lo ignorará.
    caminos_eur = [[0.0] * anios_p]
    n_sims = 1


st.subheader("💰 Amortización Extra")
with st.expander("Configurar Aportaciones"):
    cols_a = st.columns(4)
    amort_list = [cols_a[i%4].slider(f"Año {i+1}", 0, 20000, 0, step=500, key=f"a{i}") for i in range(anios_p)]

# ==========================================
# 4. CÁLCULO Y PROCESAMIENTO
# ==========================================
kpis_int, kpis_ahorro, kpis_pat = [], [], []
cuotas_matrix, eur_matrix = [], []
df_median, df_base_median = None, None

if n_sims > 50: bar = st.progress(0)

for i, camino in enumerate(caminos_eur):
    # Escenario Actual
    df = calcular_hipoteca_core(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino, amort_list, tipo_reduc)
    # Escenario Base
    df_base = calcular_hipoteca_core(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino, [0]*anios_p, 'PLAZO')
    
    # Cálculos Patrimoniales
    gasto_tot = df['Cuota'] + (s_hogar + s_vida)/12 + gastos_fijos
    df['Ahorro_Liquido'] = ahorro_inicial + (ingresos - gasto_tot).cumsum() - df['Amort_Extra'].cumsum()
    df['Equity'] = precio_vivienda - df['Saldo']
    df['Patrimonio'] = df['Ahorro_Liquido'] + df['Equity']
    
    kpis_int.append(df['Intereses'].sum())
    kpis_ahorro.append(df_base['Intereses'].sum() - df['Intereses'].sum())
    kpis_pat.append(df['Patrimonio'].iloc[-1])
    cuotas_matrix.append(df['Cuota'].values)
    eur_matrix.append(camino) # Guardamos camino Euribor para graficar
    
    if i == 0: 
        df_median = df
        df_base_median = df_base 

    if n_sims > 50: bar.progress((i+1)/n_sims)

# Seleccionar mediana
idx_med = np.argsort(kpis_int)[len(kpis_int)//2]
if n_sims > 1:
    camino_med = caminos_eur[idx_med]
    df_median = calcular_hipoteca_core(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino_med, amort_list, tipo_reduc)
    df_base_median = calcular_hipoteca_core(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino_med, [0]*anios_p, 'PLAZO')
    
    # Recalcular patrimonial mediana
    gasto_tot = df_median['Cuota'] + (s_hogar + s_vida)/12 + gastos_fijos
    df_median['Ahorro_Liquido'] = ahorro_inicial + (ingresos - gasto_tot).cumsum() - df_median['Amort_Extra'].cumsum()
    df_median['Equity'] = precio_vivienda - df_median['Saldo']
    df_median['Patrimonio'] = df_median['Ahorro_Liquido'] + df_median['Equity']

# ==========================================
# 5. DASHBOARD
# ==========================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Intereses Totales (Mediana)", f"{np.median(kpis_int):,.0f} €")
c2.metric("Ahorro Intereses", f"{np.median(kpis_ahorro):,.0f} €", delta="Generado por amortizar", delta_color="normal")
c3.metric("Patrimonio Final", f"{np.median(kpis_pat):,.0f} €")
c4.metric("Tiempo Ahorrado", f"{(len(df_base_median)-len(df_median))//12} años")

# --- PANEL DE RIESGO (Solo si hay incertidumbre) ---
if n_sims > 1 and modo_h != "FIJA":
    p5_int = np.percentile(kpis_int, 5)
    p95_int = np.percentile(kpis_int, 95)
    st.info(f"📊 **Horquilla de Riesgo (90% Probabilidad):** Pagarás entre **{p5_int:,.0f} €** (Mejor caso) y **{p95_int:,.0f} €** (Peor caso).")
elif modo_h == "FIJA":
    st.success("🔒 **Riesgo Cero:** Al ser tipo FIJO, sabes exactamente lo que vas a pagar desde el día 1.")


st.markdown("---")
# Pestañas condicionales (Ocultamos Euríbor si es Fija)
tabs = ["📉 Tipos & Cuotas", "🛡️ Estrategia Amortización", "💰 Patrimonio"]
tab1, tab2, tab3 = st.tabs(tabs)

with tab1:
    col_eur, col_cuota = st.columns(2)
    
    # 1. GRÁFICO EURIBOR (Solo si no es FIJA)
    with col_eur:
        if modo_h == "FIJA":
            st.info("Hipoteca a Tipo Fijo: Sin exposición al Euríbor.")
        else:
            st.subheader("Evolución Euríbor Previsto")
            mat_eur = np.array(eur_matrix)
            p5_eur = np.percentile(mat_eur, 5, axis=0)
            p50_eur = np.percentile(mat_eur, 50, axis=0)
            p95_eur = np.percentile(mat_eur, 95, axis=0)
            anios_x = np.arange(1, len(p50_eur)+1)
            
            fig_e = go.Figure()
            if n_sims > 1:
                fig_e.add_trace(go.Scatter(x=np.concatenate([anios_x, anios_x[::-1]]), y=np.concatenate([p95_eur, p5_eur[::-1]]), fill='toself', fillcolor='rgba(41, 128, 185, 0.2)', line=dict(color='rgba(0,0,0,0)'), name='Incertidumbre', hoverinfo="skip"))
            fig_e.add_trace(go.Scatter(x=anios_x, y=p50_eur, line=dict(color='#2980b9', width=2), name='Euríbor Mediana'))
            fig_e.update_layout(height=350, margin=dict(t=30,b=0,l=0,r=0), legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_e, use_container_width=True)

    # 2. GRÁFICO CUOTA
    with col_cuota:
        st.subheader("Evolución Cuota Mensual")
        max_len = max(len(c) for c in cuotas_matrix)
        mat_pad = np.array([np.pad(c, (0, max_len - len(c)), constant_values=np.nan) for c in cuotas_matrix])
        meses = np.arange(1, max_len + 1)
        
        fig_q = go.Figure()
        if n_sims > 1 and modo_h != "FIJA":
            p5_q = np.nanpercentile(mat_pad, 5, axis=0)
            p95_q = np.nanpercentile(mat_pad, 95, axis=0)
            fig_q.add_trace(go.Scatter(x=np.concatenate([meses, meses[::-1]]), y=np.concatenate([p95_q, p5_q[::-1]]), fill='toself', fillcolor='rgba(231,76,60,0.2)', line=dict(color='rgba(0,0,0,0)'), name='Rango P5-P95', hoverinfo="skip"))
        
        fig_q.add_trace(go.Scatter(x=meses, y=np.nanmedian(mat_pad, axis=0), line=dict(color='#c0392b', width=2), name='Cuota Mediana'))
        fig_q.update_layout(height=350, margin=dict(t=30,b=0,l=0,r=0), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_q, use_container_width=True)

with tab2:
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Interés Restante: Real vs Original")
        int_pend_base = (df_base_median['Intereses'].sum() - df_base_median['Intereses'].cumsum()).round(2)
        int_pend_real = (df_median['Intereses'].sum() - df_median['Intereses'].cumsum()).round(2)
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=df_base_median['Mes'], y=int_pend_base, name='Base', line=dict(color='gray', dash='dash')))
        fig_r.add_trace(go.Scatter(x=df_median['Mes'], y=int_pend_real, name='Real', fill='tozeroy', line=dict(color='#e74c3c')))
        fig_r.update_layout(height=350, margin=dict(t=30,b=0,l=0,r=0), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_r, use_container_width=True)
    
    with g2:
        st.subheader("Carrera Capital vs Intereses")
        fig_race = go.Figure()
        fig_race.add_trace(go.Scatter(x=df_median['Mes'], y=df_median['Capital'].cumsum(), fill='tozeroy', name='Capital', line=dict(color='#27ae60')))
        fig_race.add_trace(go.Scatter(x=df_median['Mes'], y=int_pend_real, fill='tozeroy', name='Interés', line=dict(color='#e67e22')))
        fig_race.update_layout(height=350, margin=dict(t=30,b=0,l=0,r=0), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_race, use_container_width=True)

with tab3:
    st.subheader("Evolución Patrimonio Neto")
    fig_nw = go.Figure()
    fig_nw.add_trace(go.Scatter(x=df_median['Mes'], y=df_median['Patrimonio'], name='Patrimonio Total', line=dict(color='#8e44ad', width=4)))
    fig_nw.add_trace(go.Scatter(x=df_median['Mes'], y=df_median['Equity'], name='Equity (Casa)', stackgroup='one', line=dict(color='#2980b9')))
    fig_nw.add_trace(go.Scatter(x=df_median['Mes'], y=df_median['Ahorro_Liquido'], name='Ahorro Líquido', stackgroup='one', line=dict(color='#2ecc71')))
    fig_nw.update_layout(height=400, hovermode="x unified", margin=dict(t=30,b=0,l=0,r=0))
    st.plotly_chart(fig_nw, use_container_width=True)

st.markdown("---")
with st.expander("📥 Datos Detallados (Escenario Mediana)"):
    st.dataframe(df_median.style.format("{:.2f}"))
