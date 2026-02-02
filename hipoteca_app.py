import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# CONFIGURACIÓN
# ==========================================
st.set_page_config(page_title="Simulador Hipotecario Pro 4.3 (Stable)", page_icon="🏗️", layout="wide")

# ==========================================
# 1. MOTOR MATEMÁTICO (CORE)
# ==========================================
def calcular_hipoteca_core(capital, anios, diferencial, tipo_fijo, anios_fijos, modo, euribor_puntos, amortizaciones, tipo_reduc, es_autopromotor, meses_carencia):
    n_meses_total = int(anios * 12)
    
    # Si es autopromotor, empezamos con saldo 0 (o la primera disposición)
    if es_autopromotor:
        saldo_real = 0.0
        disposicion_mensual = capital / max(1, meses_carencia) # Evitar div/0
    else:
        saldo_real = round(float(capital), 2)
        disposicion_mensual = 0
        meses_carencia = 0 
        
    saldo_teorico = round(float(capital), 2) 
    
    data = []
    mes_global = 1
    
    # Relleno seguro de listas
    puntos_eur = list(euribor_puntos) + [euribor_puntos[-1]] * (max(0, anios - len(euribor_puntos)))
    puntos_amort = list(amortizaciones) + [0] * (max(0, anios - len(amortizaciones)))

    idx_var = 0 
    
    # Bucle ANUAL
    for anio in range(anios):
        
        # 1. DETERMINAR TASA
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
        
        # 2. BUCLE MENSUAL
        for m in range(12):
            meses_restantes = n_meses_total - (mes_global - 1)
            
            # --- LÓGICA AUTOPROMOTOR (FASE CARENCIA) ---
            en_periodo_carencia = es_autopromotor and (mes_global <= meses_carencia)
            
            if en_periodo_carencia:
                # 1. Recibir disposición del mes
                saldo_real += disposicion_mensual
                if saldo_real > capital: saldo_real = capital
                
                # 2. Solo se pagan intereses sobre lo dispuesto
                cuota = saldo_real * tasa_mensual
                interes_m = cuota
                capital_m = 0 
                
            else:
                # --- LÓGICA NORMAL ---
                # Al terminar la carencia, aseguramos que el saldo es el capital total
                if es_autopromotor and mes_global == meses_carencia + 1:
                     saldo_real = round(float(capital), 2)
                
                if saldo_real <= 0.01:
                    saldo_real = 0
                    cuota = 0
                    interes_m = 0
                    capital_m = 0
                else:
                    base_calc = saldo_teorico if tipo_reduc == 'PLAZO' else saldo_real
                    if base_calc < saldo_real: base_calc = saldo_real
                    
                    if tasa_mensual > 0:
                        try:
                            cuota = base_calc * (tasa_mensual * (1 + tasa_mensual)**meses_restantes) / ((1 + tasa_mensual)**meses_restantes - 1)
                        except:
                            cuota = base_calc / meses_restantes
                    else:
                        cuota = base_calc / meses_restantes
                    
                    cuota = round(cuota, 2)
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

            # Guardar datos del mes
            data.append({
                'Mes': mes_global, 'Año': anio + 1, 'Tasa': tasa_anual if saldo_real > 0 else 0, 
                'Cuota': cuota, 'Intereses': interes_m, 'Capital': capital_m, 
                'Saldo': saldo_real, 'Amort_Extra': 0,
                'Fase': 'Carencia' if en_periodo_carencia else 'Amortización'
            })
            
            # 3. AMORTIZACIÓN EXTRA (No permitida en fase carencia)
            if not en_periodo_carencia:
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
# 2. INTERFAZ DINÁMICA
# ==========================================
st.title("🏗️ Simulador Hipotecario Pro 4.3")
st.markdown("---")

with st.sidebar:
    st.header("👤 Perfil Financiero")
    ingresos = st.number_input("Ingresos Mensuales (€)", value=2500, step=100)
    ahorro_inicial = st.number_input("Ahorro Inicial (€)", value=0, step=1000)
    precio_vivienda = st.number_input("Valor Vivienda/Obra (€)", value=0, step=5000)
    
    st.markdown("---")
    st.header("⚙️ Estructura Préstamo")
    
    # DATOS COMUNES
    modo_h = st.selectbox("Modalidad", ["MIXTA", "VARIABLE", "FIJA"])
    
    # --- OPCIÓN AUTOPROMOTOR ---
    es_autopromotor = st.checkbox("🏗️ Es Autopromoción (Obra Nueva)", value=False)
    meses_carencia = 0
    if es_autopromotor:
        meses_carencia = st.number_input("Meses Carencia (Construcción)", value=11, min_value=1, max_value=36)
        st.info(f"Durante los primeros {meses_carencia} meses pagarás solo intereses sobre el dinero dispuesto.")
    # ---------------------------
    
    capital_init = st.number_input("Capital Pendiente / A Solicitar (€)", value=180000, step=1000)
    anios_p = st.number_input("Años Totales (Incluye carencia)", value=25, min_value=1)
    tipo_reduc = st.radio("Estrategia Amortización", ["PLAZO", "CUOTA"])
    
    st.markdown("---")
    st.header("🏦 Condiciones Banco")
    
    tipo_fijo = 0.0
    diferencial = 0.0
    anios_fijos = 0
    
    if modo_h == "FIJA":
        tipo_fijo = st.number_input("Tipo Fijo (%)", value=2.50, step=0.05)
    elif modo_h == "VARIABLE":
        diferencial = st.number_input("Diferencial (%)", value=0.55, step=0.05)
    elif modo_h == "MIXTA":
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            tipo_fijo = st.number_input("Tipo Fijo (%)", value=2.25, step=0.05)
        with col_m2:
            anios_fijos = st.number_input("Años Fijos", value=5, min_value=1, max_value=anios_p-1)
        diferencial = st.number_input("Diferencial (%)", value=0.55, step=0.05)

    st.markdown("---")
    st.header("🖇️ Vinculaciones (Seguros)")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        s_hogar = st.number_input("Seguro Hogar (€/año)", value=300)
    with col_s2:
        s_vida = st.number_input("Seguro Vida (€/año)", value=400)
    
    st.markdown("---")
    st.header("🛡️ Gastos de Vida")
    g_comida = st.number_input("Comida (€)", value=400, step=50)
    g_suministros = st.number_input("Suministros (€)", value=150, step=10)
    g_gasolina = st.number_input("Transporte (€)", value=100, step=10)
    g_otros = st.number_input("Otros gastos (€)", value=200, step=10)


# ==========================================
# 3. CONFIGURACIÓN EURÍBOR
# ==========================================
caminos_eur = []
modo_prev = "N/A"
n_sims = 1

if modo_h != "FIJA":
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

    n_años_var = anios_p if modo_h == "VARIABLE" else max(0, anios_p - anios_fijos)
    
    if n_años_var > 0:
        if modo_prev == "Manual (Sliders)":
            with st.expander("🛠️ Ajustar Euríbor Manualmente", expanded=True):
                # CORRECCIÓN: Se suma 'anios_fijos' al índice para reflejar el año real (ej: A6, A7...)
                eur_list = [st.slider(f"A{i + 1 + anios_fijos}", -1.0, 7.0, 2.25, key=f"e{i}") for i in range(n_años_var)]
            caminos_eur = [eur_list]
        else:
            caminos_eur = simular_vasicek(r0, theta, kappa, sigma, n_años_var, n_sims)
else:
    caminos_eur = [[0.0] * anios_p]
    n_sims = 1


st.subheader("💰 Amortización Extra")
with st.expander("Configurar Aportaciones Anuales"):
    cols_a = st.columns(4)
    amort_list = [cols_a[i%4].slider(f"Año {i+1}", 0, 20000, 0, step=500, key=f"a{i}") for i in range(anios_p)]

# ==========================================
# 4. CÁLCULO Y PROCESAMIENTO
# ==========================================
kpis_int, kpis_ahorro, kpis_pat, kpis_seguros = [], [], [], []
cuotas_matrix, eur_matrix = [], []
df_median, df_base_median = None, None

if n_sims > 50: bar = st.progress(0)

total_gastos_vida_mensual = g_comida + g_suministros + g_gasolina + g_otros
coste_mensual_seguros = (s_hogar + s_vida) / 12

for i, camino in enumerate(caminos_eur):
    # Escenario Actual
    df = calcular_hipoteca_core(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino, amort_list, tipo_reduc, es_autopromotor, meses_carencia)
    # Escenario Base
    df_base = calcular_hipoteca_core(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino, [0]*anios_p, 'PLAZO', es_autopromotor, meses_carencia)
    
    # --- CÁLCULOS PATRIMONIALES ---
    df['Seguros_Pagados'] = np.where(df['Saldo'] > 0, coste_mensual_seguros, 0)
    gasto_tot = df['Cuota'] + df['Seguros_Pagados'] + total_gastos_vida_mensual
    
    df['Ahorro_Liquido'] = ahorro_inicial + (ingresos - gasto_tot).cumsum() - df['Amort_Extra'].cumsum()
    df['Equity'] = precio_vivienda - df['Saldo']
    df['Patrimonio'] = df['Ahorro_Liquido'] + df['Equity']
    
    kpis_int.append(df['Intereses'].sum())
    kpis_seguros.append(df['Seguros_Pagados'].sum())
    kpis_ahorro.append(df_base['Intereses'].sum() - df['Intereses'].sum())
    kpis_pat.append(df['Patrimonio'].iloc[-1])
    
    cuotas_matrix.append(df['Cuota'].values)
    eur_matrix.append(camino)
    
    if i == 0: 
        df_median = df
        df_base_median = df_base 

    if n_sims > 50: bar.progress((i+1)/n_sims)

# Seleccionar escenario mediana
idx_med = np.argsort(kpis_int)[len(kpis_int)//2]
if n_sims > 1:
    camino_med = caminos_eur[idx_med]
    df_median = calcular_hipoteca_core(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino_med, amort_list, tipo_reduc, es_autopromotor, meses_carencia)
    df_base_median = calcular_hipoteca_core(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino_med, [0]*anios_p, 'PLAZO', es_autopromotor, meses_carencia)
    
    # Recalcular patrimonial mediana
    df_median['Seguros_Pagados'] = np.where(df_median['Saldo'] > 0, coste_mensual_seguros, 0)
    gasto_tot = df_median['Cuota'] + df_median['Seguros_Pagados'] + total_gastos_vida_mensual
    df_median['Ahorro_Liquido'] = ahorro_inicial + (ingresos - gasto_tot).cumsum() - df_median['Amort_Extra'].cumsum()
    df_median['Equity'] = precio_vivienda - df_median['Saldo']
    df_median['Patrimonio'] = df_median['Ahorro_Liquido'] + df_median['Equity']

# ==========================================
# 5. DASHBOARD
# ==========================================
intereses_totales = np.median(kpis_int)
seguros_totales = np.median(kpis_seguros)
coste_total_operacion = intereses_totales + seguros_totales
ahorro_intereses = np.median(kpis_ahorro)
meses_con_hipoteca = len(df_median[df_median['Saldo'] > 0])
meses_ahorrados = (anios_p * 12) - meses_con_hipoteca

st.subheader("📊 Análisis de Costes")
c1, c2, c3 = st.columns(3)
c1.metric("Intereses Banco", f"{intereses_totales:,.0f} €")
c2.metric("Gasto en Seguros", f"{seguros_totales:,.0f} €")
c3.metric("COSTE REAL TOTAL", f"{coste_total_operacion:,.0f} €", delta="Intereses + Seguros", delta_color="off")

st.markdown("---")
st.subheader("🚀 Beneficio por Amortizar")
c4, c5, c6 = st.columns(3)
c4.metric("Ahorro Intereses", f"{ahorro_intereses:,.0f} €", delta="Generado por amortizar", delta_color="normal")
c5.metric("Tiempo Ahorrado", f"{meses_ahorrados // 12} a, {meses_ahorrados % 12} m")
c6.metric("Patrimonio Final (Año 25)", f"{np.median(kpis_pat):,.0f} €", help="Incluye todo el ahorro acumulado", delta="Valor Real Comparable")

# --- PANEL DE RIESGO ---
st.markdown("---")
if n_sims > 1 and modo_h != "FIJA":
    p5_int = np.percentile(kpis_int, 5)
    p95_int = np.percentile(kpis_int, 95)
    p10_int = np.percentile(kpis_int, 10)
    p90_int = np.percentile(kpis_int, 90)
    
    st.subheader("🎲 Riesgo de Intereses (Monte Carlo)")
    cr1, cr2 = st.columns(2)
    with cr1:
        st.info(f"📊 **Horquilla 90% Probabilidad:**\nPagarás entre **{p5_int:,.0f} €** y **{p95_int:,.0f} €** de intereses.")
    with cr2:
        st.warning(f"🎯 **Horquilla 80% Probabilidad:**\nPagarás entre **{p10_int:,.0f} €** y **{p90_int:,.0f} €** de intereses.")
elif modo_h == "FIJA":
    st.success("🔒 **Riesgo Cero:** Al ser tipo FIJO, sabes exactamente lo que vas a pagar desde el día 1.")

st.markdown("---")
tabs = ["📉 Tipos & Cuotas", "🛡️ Estrategia Amortización", "💰 Patrimonio"]
tab1, tab2, tab3 = st.tabs(tabs)

with tab1:
    col_eur, col_cuota = st.columns(2)
    with col_eur:
        if modo_h == "FIJA":
            st.info("Hipoteca a Tipo Fijo: Sin exposición al Euríbor.")
        else:
            st.subheader("Evolución Euríbor Previsto")
            mat_eur = np.array(eur_matrix)
            p50_eur = np.percentile(mat_eur, 50, axis=0)
            anios_x = np.arange(1, len(p50_eur)+1)
            fig_e = go.Figure()
            fig_e.add_trace(go.Scatter(x=anios_x, y=p50_eur, line=dict(color='#2980b9', width=2), name='Euríbor Mediana'))
            fig_e.update_layout(height=350, margin=dict(t=30,b=0,l=0,r=0))
            st.plotly_chart(fig_e, use_container_width=True)

    with col_cuota:
        st.subheader("Evolución Cuota Mensual")
        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(x=df_median['Mes'], y=df_median['Cuota'], line=dict(color='#c0392b', width=2), name='Cuota Pagada'))
        if es_autopromotor:
            fig_q.add_vline(x=meses_carencia, line_dash="dash", line_color="green", annotation_text="Fin Carencia")
        fig_q.update_layout(height=350, margin=dict(t=30,b=0,l=0,r=0))
        st.plotly_chart(fig_q, use_container_width=True)

with tab2:
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Interés Acumulado")
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=df_base_median['Mes'], y=df_base_median['Intereses'].cumsum(), name='Base', line=dict(color='gray', dash='dash')))
        fig_r.add_trace(go.Scatter(x=df_median['Mes'], y=df_median['Intereses'].cumsum(), name='Amortizando', line=dict(color='#e74c3c')))
        fig_r.update_layout(height=350, margin=dict(t=30,b=0,l=0,r=0), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_r, use_container_width=True)
    
    with g2:
        st.subheader("Deuda Pendiente")
        fig_race = go.Figure()
        fig_race.add_trace(go.Scatter(x=df_base_median['Saldo'], name='Saldo Base', line=dict(color='gray', dash='dash'))) 
        fig_race.add_trace(go.Scatter(x=df_median['Saldo'], fill='tozeroy', name='Saldo Real', line=dict(color='#27ae60')))
        fig_race.update_layout(height=350, margin=dict(t=30,b=0,l=0,r=0), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_race, use_container_width=True)

with tab3:
    st.subheader("Evolución Patrimonio Neto")
    fig_nw = go.Figure()
    
    fig_nw.add_trace(go.Scatter(x=df_median['Mes'], y=df_median['Patrimonio'], name='Patrimonio (Amortizando)', line=dict(color='#8e44ad', width=4)))
    
    gasto_base = df_base_median['Cuota'] + coste_mensual_seguros + total_gastos_vida_mensual
    ahorro_base = ahorro_inicial + (ingresos - gasto_base).cumsum()
    pat_base = ahorro_base + (precio_vivienda - df_base_median['Saldo'])
    
    fig_nw.add_trace(go.Scatter(x=df_base_median['Mes'], y=pat_base, name='Patrimonio (Base)', line=dict(color='gray', dash='dot')))
    
    fig_nw.update_layout(height=400, hovermode="x unified", margin=dict(t=30,b=0,l=0,r=0))
    st.plotly_chart(fig_nw, use_container_width=True)

st.markdown("---")
with st.expander("📥 Datos Detallados (Escenario Mediana)"):
    # Fix: Filtramos solo las columnas numéricas para el formateo para no romper con la columna de texto 'Fase'
    numeric_cols = df_median.select_dtypes(include=[np.number]).columns
    st.dataframe(df_median.style.format("{:.2f}", subset=numeric_cols))
