import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# CONFIGURACIÓN DE PÁGINA CORPORATIVA
# ==========================================
st.set_page_config(
    page_title="Terminal Financiero - Hipotecas", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS: ESTILO INSTITUCIONAL/BANCA ---
st.markdown("""
<style>
    /* Tipografía Corporativa */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #333333;
    }
    
    /* Cabeceras sobrias */
    h1, h2, h3, h4, h5 {
        color: #002B5B; /* Navy Blue */
        font-weight: 600;
        letter-spacing: -0.5px;
        text-transform: uppercase;
    }
    
    /* Métricas estilo Bloomberg/Reuters */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border-left: 4px solid #002B5B;
        padding: 10px 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    label[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: #666;
        text-transform: uppercase;
        font-weight: bold;
    }
    
    /* Sidebar Profesional */
    section[data-testid="stSidebar"] {
        background-color: #f4f6f9;
        border-right: 1px solid #d1d9e6;
    }
    
    /* Ajustes de Inputs */
    .stNumberInput, .stSelectbox, .stCheckbox {
        font-size: 0.9rem;
    }
    
    /* Ocultar elementos de UI decorativos de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. MOTOR MATEMÁTICO (CORE)
# ==========================================
def calcular_hipoteca_core(capital, anios, diferencial, tipo_fijo, anios_fijos, modo, euribor_puntos, amortizaciones, tipo_reduc, es_autopromotor, meses_carencia):
    n_meses_total = int(anios * 12)
    
    if es_autopromotor:
        saldo_real = 0.0
        disposicion_mensual = capital / max(1, meses_carencia)
    else:
        saldo_real = round(float(capital), 2)
        disposicion_mensual = 0
        meses_carencia = 0 
        
    saldo_teorico = round(float(capital), 2) 
    
    data = []
    mes_global = 1
    
    # Normalización de vectores
    puntos_eur = list(euribor_puntos) + [euribor_puntos[-1]] * (max(0, anios - len(euribor_puntos)))
    puntos_amort = list(amortizaciones) + [0] * (max(0, anios - len(amortizaciones)))

    idx_var = 0 
    
    for anio in range(anios):
        # Determinación de tasa anual
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
        
        for m in range(12):
            meses_restantes = n_meses_total - (mes_global - 1)
            en_periodo_carencia = es_autopromotor and (mes_global <= meses_carencia)
            
            if en_periodo_carencia:
                saldo_real += disposicion_mensual
                if saldo_real > capital: saldo_real = capital
                cuota = saldo_real * tasa_mensual
                interes_m = cuota
                capital_m = 0 
            else:
                if es_autopromotor and mes_global == meses_carencia + 1:
                      saldo_real = round(float(capital), 2)
                
                if saldo_real <= 0.01:
                    saldo_real = 0; cuota = 0; interes_m = 0; capital_m = 0
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
                    
                    # Cálculo teórico para reducción de plazo
                    int_teorico = round(saldo_teorico * tasa_mensual, 2)
                    amort_teorica = round(cuota - int_teorico, 2)
                    saldo_teorico = round(saldo_teorico - amort_teorica, 2)
                    if saldo_teorico < 0: saldo_teorico = 0

            data.append({
                'Mes': mes_global, 'Año': anio + 1, 'Tasa': tasa_anual if saldo_real > 0 else 0, 
                'Cuota': cuota, 'Intereses': interes_m, 'Capital': capital_m, 
                'Saldo': saldo_real, 'Amort_Extra': 0,
                'Fase': 'Carencia' if en_periodo_carencia else 'Amortización'
            })
            
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
    np.random.seed(42) # Semilla fija para consistencia
    for _ in range(n_sims):
        camino = [r0]
        for t in range(anios - 1):
            dr = kappa * (theta - camino[-1]) * dt + sigma * np.random.normal()
            camino.append(max(-1.0, camino[-1] + dr))
        sims.append(camino)
    return np.array(sims)

# ==========================================
# 2. PANEL LATERAL DE CONTROL
# ==========================================
st.markdown("### ANÁLISIS FINANCIERO DE PRÉSTAMOS")
st.markdown("---")

with st.sidebar:
    st.markdown("#### PERFIL DE CLIENTE")
    ingresos = st.number_input("Ingresos Netos Mensuales (€)", value=2500, step=100)
    ahorro_inicial = st.number_input("Fondos Propios Disponibles (€)", value=0, step=1000)
    precio_vivienda = st.number_input("Valor de Tasación (€)", value=0, step=5000)
    
    st.markdown("---")
    st.markdown("#### ESTRUCTURA OPERACIÓN")
    
    # MIXTA POR DEFECTO (Index 0)
    modo_h = st.selectbox("Modalidad Tipo Interés", ["MIXTA", "VARIABLE", "FIJA"], index=0)
    
    # AUTOPROMOCIÓN POR DEFECTO
    es_autopromotor = st.checkbox("Promoción/Autoconstrucción", value=True)
    
    meses_carencia = 0
    if es_autopromotor:
        meses_carencia = st.number_input("Periodo Carencia (Meses)", value=11, min_value=1, max_value=36)
        
    capital_init = st.number_input("Importe Préstamo (€)", value=180000, step=1000)
    anios_p = st.number_input("Plazo Amortización (Años)", value=25, min_value=1)
    tipo_reduc = st.selectbox("Destino Amortización Anticipada", ["REDUCCIÓN DE PLAZO", "REDUCCIÓN DE CUOTA"])
    
    st.markdown("---")
    st.markdown("#### CONDICIONES BANCARIAS")
    
    # Lógica para mostrar inputs según el modo seleccionado
    tipo_fijo = 0.0
    diferencial = 0.0
    anios_fijos = 0
    
    if modo_h == "FIJA":
        tipo_fijo = st.number_input("TIN Fijo (%)", value=2.75, step=0.05)
    elif modo_h == "VARIABLE":
        diferencial = st.number_input("Diferencial s/Euribor (%)", value=0.55, step=0.05)
    elif modo_h == "MIXTA":
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            # 2.25 POR DEFECTO
            tipo_fijo = st.number_input("TIN Tramo Fijo (%)", value=2.25, step=0.05)
        with col_m2:
            # 7 AÑOS POR DEFECTO
            anios_fijos = st.number_input("Duración Fijo (Años)", value=7, min_value=1, max_value=anios_p-1)
        # 0.55 POR DEFECTO
        diferencial = st.number_input("Diferencial Variable (%)", value=0.55, step=0.05)

    st.markdown("---")
    st.markdown("#### GASTOS OPERATIVOS")
    s_hogar = st.number_input("Prima Seg. Hogar (Anual)", value=300)
    s_vida = st.number_input("Prima Seg. Vida (Anual)", value=300)
    
    # Calculo del default sumando los gastos desglosados
    # Alimentación 300 + Suministros 150 + Gasolina 120 + Subs 75 + Otros 150 = 795
    default_gastos = 300 + 150 + 120 + 75 + 150
    total_gastos_vida_mensual = st.number_input("Gastos Recurrentes (Total)", value=default_gastos, help="Suma de alimentación, suministros, transporte, suscripciones y otros.")

# ==========================================
# 3. ESCENARIOS DE RIESGO
# ==========================================
caminos_eur = []
n_sims = 1

if modo_h != "FIJA":
    with st.sidebar:
        st.markdown("---")
        st.markdown("#### PARAMETRIZACIÓN RIESGO")
        
        sim_mode = st.radio("Modelo de Tipos", ["Estocástico (Monte Carlo)", "Curva Manual"])
        
        if sim_mode == "Estocástico (Monte Carlo)":
            n_sims = st.select_slider("Iteraciones", options=[50, 100, 500, 1000], value=100)
            theta = st.slider("Media Reversión (θ)", 0.0, 5.0, 2.50)
            sigma = st.slider("Volatilidad (σ)", 0.0, 2.0, 0.50)
            kappa = st.slider("Velocidad Ajuste (κ)", 0.0, 1.0, 0.20)
            r0 = st.number_input("Euríbor Spot (%)", value=2.50)
        else:
            n_sims = 1

    n_años_var = anios_p if modo_h == "VARIABLE" else max(0, anios_p - anios_fijos)
    
    if n_años_var > 0:
        if sim_mode == "Curva Manual":
            with st.expander("Definición Curva Forward"):
                eur_list = []
                cols_man = st.columns(3)
                for i in range(n_años_var):
                    val = cols_man[i%3].number_input(f"Y{i + 1 + anios_fijos}", -1.0, 10.0, 2.5, step=0.1, key=f"e{i}")
                    eur_list.append(val)
            caminos_eur = [eur_list]
        else:
            caminos_eur = simular_vasicek(r0, theta, kappa, sigma, n_años_var, n_sims)
else:
    caminos_eur = [[0.0] * anios_p]
    n_sims = 1

with st.expander("PLANIFICACIÓN DE APORTACIONES EXTRAORDINARIAS", expanded=False):
    # Generamos dinámicamente una columna por cada año del préstamo
    cols_a = st.columns(anios_p)
    amort_list = []
    
    for i in range(anios_p):
        with cols_a[i]:
            # Indicador del año compacto (A1, A2...) para ahorrar espacio
            st.markdown(f"<div style='text-align:center; font-size:10px; color:#666;'>A{i+1}</div>", unsafe_allow_html=True)
            val = st.slider(
                label=f"Año {i+1+anios_fijos}",
                min_value=0, 
                max_value=10000, 
                value=0, 
                step=1000, 
                key=f"a{i}",
                label_visibility="collapsed" # Ocultamos la etiqueta estándar para que quepan
            )
            amort_list.append(val)

# ==========================================
# 4. PROCESAMIENTO
# ==========================================
kpis_int, kpis_ahorro, kpis_pat, kpis_seguros = [], [], [], []
cuotas_matrix, eur_matrix = [], []
df_median, df_base_median = None, None

coste_mensual_seguros = (s_hogar + s_vida) / 12

if n_sims > 100: 
    progress_bar = st.progress(0)

for i, camino in enumerate(caminos_eur):
    # Escenario Actual
    df = calcular_hipoteca_core(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino, amort_list, tipo_reduc, es_autopromotor, meses_carencia)
    # Escenario Base
    df_base = calcular_hipoteca_core(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino, [0]*anios_p, 'REDUCCIÓN DE PLAZO', es_autopromotor, meses_carencia)
    
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
    
    if n_sims > 100: progress_bar.progress((i+1)/n_sims)

# Selección de Mediana
idx_med = np.argsort(kpis_int)[len(kpis_int)//2]
if n_sims > 1:
    camino_med = caminos_eur[idx_med]
    df_median = calcular_hipoteca_core(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino_med, amort_list, tipo_reduc, es_autopromotor, meses_carencia)
    df_base_median = calcular_hipoteca_core(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino_med, [0]*anios_p, 'REDUCCIÓN DE PLAZO', es_autopromotor, meses_carencia)
    
    df_median['Seguros_Pagados'] = np.where(df_median['Saldo'] > 0, coste_mensual_seguros, 0)
    gasto_tot = df_median['Cuota'] + df_median['Seguros_Pagados'] + total_gastos_vida_mensual
    df_median['Ahorro_Liquido'] = ahorro_inicial + (ingresos - gasto_tot).cumsum() - df_median['Amort_Extra'].cumsum()
    df_median['Equity'] = precio_vivienda - df_median['Saldo']
    df_median['Patrimonio'] = df_median['Ahorro_Liquido'] + df_median['Equity']
else:
    df_median = df
    df_base_median = df_base

# ==========================================
# 5. DASHBOARD
# ==========================================

intereses_totales = np.median(kpis_int)
seguros_totales = np.median(kpis_seguros)
coste_total_operacion = intereses_totales + seguros_totales
ahorro_intereses = np.median(kpis_ahorro)

c1, c2, c3, c4 = st.columns(4)
c1.metric("COSTE FINANCIERO", f"{intereses_totales:,.2f} €")
c2.metric("COSTE SEGUROS", f"{seguros_totales:,.2f} €")
c3.metric("COSTE TOTAL", f"{coste_total_operacion:,.2f} €")
c4.metric("AHORRO POTENCIAL", f"{ahorro_intereses:,.2f} €")

st.markdown("### INFORME DETALLADO")

tabs = st.tabs(["POSICIÓN GLOBAL", "INCERTIDUMBRE (RIESGO)", "DETALLE AMORTIZACIÓN", "TABLA DATOS"])

# Layout común para gráficos Plotly
layout_corp = dict(
    font=dict(family="Arial", size=11, color="#333"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=40, r=40, t=30, b=30),
    xaxis=dict(showgrid=True, gridcolor="#eee", linecolor="#333"),
    yaxis=dict(showgrid=True, gridcolor="#eee", linecolor="#333"),
    legend=dict(orientation="h", y=1.1, x=0)
)

# -----------------------------------------------------
# TAB 1: POSICIÓN GLOBAL (MODIFICADO)
# -----------------------------------------------------
with tabs[0]:
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("**PREVISIÓN EURÍBOR (ESCENARIO MEDIANA)**")
        
        # Lógica de visualización Euríbor
        fig_e = go.Figure()
        
        if modo_h == "FIJA":
             st.info("OPERACIÓN A TIPO FIJO. SIN EXPOSICIÓN AL EURÍBOR.")
        else:
            # Si hay simulación, mostramos el Fan Chart aquí para dar contexto "Global"
            # O solo la línea mediana si se prefiere limpieza. Usaremos Fan Chart limpio.
            
            mat_eur = np.array(eur_matrix)
            p50 = np.percentile(mat_eur, 50, axis=0)
            x_axis = np.arange(1, len(p50) + 1)
            
            if n_sims > 1:
                p25 = np.percentile(mat_eur, 25, axis=0)
                p75 = np.percentile(mat_eur, 75, axis=0)
                
                fig_e.add_trace(go.Scatter(
                    x=np.concatenate([x_axis, x_axis[::-1]]),
                    y=np.concatenate([p75, p25[::-1]]),
                    fill='toself', fillcolor='rgba(0, 43, 91, 0.1)',
                    line=dict(color='rgba(255,255,255,0)'), name='Rango Probable'
                ))
            
            fig_e.add_trace(go.Scatter(
                x=x_axis, y=p50,
                line=dict(color='#002B5B', width=2), name='Euríbor Estimado'
            ))
            
            fig_e.update_layout(height=350, yaxis_title="Tipo (%)", **layout_corp)
            st.plotly_chart(fig_e, use_container_width=True)
        
    with col_g2:
        st.markdown("**CUOTA MENSUAL ESTIMADA**")
        fig_q = go.Figure()
        
        fig_q.add_trace(go.Scatter(
            x=df_median['Mes'], y=df_median['Cuota'],
            line=dict(color='#800000', width=2), name='Cuota Mensual'
        ))
        
        if es_autopromotor:
            fig_q.add_vline(x=meses_carencia, line_dash="dash", line_color="grey", annotation_text="FIN CARENCIA")
            
        fig_q.update_layout(height=350, yaxis_title="Cuota (€)", **layout_corp)
        st.plotly_chart(fig_q, use_container_width=True)

# -----------------------------------------------------
# TAB 2: INCERTIDUMBRE (RIESGO)
# -----------------------------------------------------
with tabs[1]:
    if modo_h == "FIJA":
        st.info("EL ANÁLISIS DE INCERTIDUMBRE NO APLICA A PRÉSTAMOS A TIPO FIJO.")
    else:
        c_risk1, c_risk2 = st.columns([2, 1])
        
        with c_risk1:
            st.markdown("**DISTRIBUCIÓN DE PROBABILIDAD DE TIPOS (FAN CHART)**")
            
            mat_eur = np.array(eur_matrix)
            p5 = np.percentile(mat_eur, 5, axis=0)
            p95 = np.percentile(mat_eur, 95, axis=0)
            p50 = np.percentile(mat_eur, 50, axis=0)
            x_axis = np.arange(1, len(p50) + 1)
            
            fig_fan = go.Figure()
            
            # 90%
            fig_fan.add_trace(go.Scatter(
                x=np.concatenate([x_axis, x_axis[::-1]]),
                y=np.concatenate([p95, p5[::-1]]),
                fill='toself', fillcolor='rgba(173, 216, 230, 0.3)',
                line=dict(color='rgba(255,255,255,0)'), name='Intervalo 90%'
            ))
            
            # Mediana
            fig_fan.add_trace(go.Scatter(
                x=x_axis, y=p50,
                line=dict(color='#002B5B', width=2), name='Escenario Central'
            ))
            
            fig_fan.update_layout(height=400, yaxis_title="Euríbor (%)", **layout_corp)
            st.plotly_chart(fig_fan, use_container_width=True)
            
        with c_risk2:
            st.markdown("**IMPACTO EN COSTE FINANCIERO**")
            
            p5_int = np.percentile(kpis_int, 5)
            p95_int = np.percentile(kpis_int, 95)
            
            st.write(f"**Mejor Caso (P5):** {p5_int:,.0f} €")
            st.write(f"**Caso Central:** {intereses_totales:,.0f} €")
            st.write(f"**Peor Caso (P95):** {p95_int:,.0f} €")
            
            st.markdown("---")
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=kpis_int, 
                marker_color='#002B5B',
                opacity=0.7,
                nbinsx=20,
                name='Frecuencia'
            ))
            
            # Corrección de TypeError usando update sobre copia
            hist_layout = layout_corp.copy()
            hist_layout.update(dict(
                height=250, 
                margin=dict(l=20, r=20, t=10, b=20),
                showlegend=False
            ))
            
            fig_hist.update_layout(**hist_layout)
            st.plotly_chart(fig_hist, use_container_width=True)

# -----------------------------------------------------
# TAB 3: DETALLE AMORTIZACIÓN
# -----------------------------------------------------
with tabs[2]:
    col_c, col_i = st.columns(2)
    with col_c:
        st.markdown("**EVOLUCIÓN SALDO VIVO**")
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=df_base_median['Mes'], y=df_base_median['Saldo'],
            name='Cuadro Base', line=dict(color='#B0B0B0', width=2, dash='dot')
        ))
        fig_s.add_trace(go.Scatter(
            x=df_median['Mes'], y=df_median['Saldo'],
            name='Con Amortización', line=dict(color='#002B5B', width=3),
            fill='tozeroy', fillcolor='rgba(0, 43, 91, 0.1)'
        ))
        fig_s.update_layout(height=350, **layout_corp)
        st.plotly_chart(fig_s, use_container_width=True)
        
    with col_i:
        st.markdown("**INTERÉS ACUMULADO**")
        fig_ia = go.Figure()
        fig_ia.add_trace(go.Scatter(
            x=df_base_median['Mes'], y=df_base_median['Intereses'].cumsum(),
            name='Base', line=dict(color='#B0B0B0', dash='dash')
        ))
        fig_ia.add_trace(go.Scatter(
            x=df_median['Mes'], y=df_median['Intereses'].cumsum(),
            name='Optimizado', line=dict(color='#002B5B')
        ))
        fig_ia.update_layout(height=350, **layout_corp)
        st.plotly_chart(fig_ia, use_container_width=True)

# -----------------------------------------------------
# TAB 4: TABLA DE DATOS
# -----------------------------------------------------
with tabs[3]:
    st.markdown("**CUADRO DE AMORTIZACIÓN (ESCENARIO CENTRAL)**")
    numeric_cols = df_median.select_dtypes(include=[np.number]).columns
    st.dataframe(
        df_median.style.format("{:.2f}", subset=numeric_cols),
        use_container_width=True,
        height=500
    )
