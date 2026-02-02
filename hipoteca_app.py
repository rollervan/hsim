import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ==========================================
st.set_page_config(
    page_title="Simulador Financiero Hipotecario",
    page_icon="bar_chart",
    layout="wide"
)

# CSS para 'limpiar' la interfaz y dar aspecto profesional
st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    h1 {font-family: 'Helvetica', sans-serif; font-weight: 700; color: #2c3e50;}
    h2, h3 {font-family: 'Helvetica', sans-serif; color: #34495e;}
    .stMetric {background-color: #f8f9fa; border: 1px solid #e9ecef; padding: 10px; border-radius: 5px;}
    .stMetric label {color: #6c757d;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. MOTOR MATEMÁTICO (CORE) - INTACTO
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
    
    puntos_eur = list(euribor_puntos) + [euribor_puntos[-1]] * (max(0, anios - len(euribor_puntos)))
    puntos_amort = list(amortizaciones) + [0] * (max(0, anios - len(amortizaciones)))

    idx_var = 0 
    
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
                saldo_real += disposicion_mensual
                if saldo_real > capital: saldo_real = capital
                
                cuota = saldo_real * tasa_mensual
                interes_m = cuota
                capital_m = 0 
                
            else:
                # --- LÓGICA NORMAL ---
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
# 2. INTERFAZ: SIDEBAR (INPUTS)
# ==========================================
with st.sidebar:
    st.markdown("### Parámetros del Proyecto")
    
    # Sección 1: Datos Personales
    with st.expander("Perfil Económico", expanded=True):
        ingresos = st.number_input("Ingresos netos mensuales (€)", value=2500, step=100)
        ahorro_inicial = st.number_input("Ahorro previo disponible (€)", value=0, step=1000)
        precio_vivienda = st.number_input("Valor de compra / tasación (€)", value=0, step=5000)

    # Sección 2: Estructura del Préstamo
    st.markdown("---")
    st.markdown("**Configuración Hipoteca**")
    
    modo_h = st.selectbox("Modalidad de Tipo", ["MIXTA", "VARIABLE", "FIJA"])
    
    es_autopromotor = st.checkbox("Préstamo Autopromotor (Obra Nueva)", value=False)
    meses_carencia = 0
    if es_autopromotor:
        meses_carencia = st.number_input("Meses de carencia técnica", value=11, min_value=1, max_value=36, help="Meses donde solo se pagan intereses mientras se construye.")
    
    capital_init = st.number_input("Capital Solicitado (€)", value=180000, step=1000)
    anios_p = st.number_input("Duración (Años)", value=25, min_value=1)
    
    # Sección 3: Condiciones Bancarias
    st.markdown("---")
    st.markdown("**Condiciones Financieras**")
    
    tipo_fijo = 0.0
    diferencial = 0.0
    anios_fijos = 0
    
    col_cond1, col_cond2 = st.columns(2)
    if modo_h == "FIJA":
        with col_cond1:
            tipo_fijo = st.number_input("TIN Fijo (%)", value=2.50, step=0.05)
    elif modo_h == "VARIABLE":
        with col_cond1:
            diferencial = st.number_input("Diferencial (%)", value=0.55, step=0.05)
    elif modo_h == "MIXTA":
        with col_cond1:
            tipo_fijo = st.number_input("TIN Tramo Fijo (%)", value=2.25, step=0.05)
            anios_fijos = st.number_input("Duración Fijo (años)", value=5, min_value=1, max_value=anios_p-1)
        with col_cond2:
            diferencial = st.number_input("Diferencial Variable (%)", value=0.55, step=0.05)

    tipo_reduc = st.radio("Estrategia de Amortización", ["Reducción de PLAZO", "Reducción de CUOTA"], index=0)

    # Sección 4: Gastos y Seguros
    with st.expander("Vinculaciones y Gastos de Vida", expanded=False):
        st.caption("Seguros anuales vinculados")
        c_seg1, c_seg2 = st.columns(2)
        s_hogar = c_seg1.number_input("Hogar (€/año)", value=300)
        s_vida = c_seg2.number_input("Vida (€/año)", value=300)
        
        st.caption("Gastos mensuales recurrentes (para cálculo de ahorro)")
        g_comida = st.number_input("Alimentación", value=300, step=50)
        g_suministros = st.number_input("Suministros (Luz/Agua/Net)", value=150, step=10)
        g_gasolina = st.number_input("Transporte", value=100, step=10)
        g_otros = st.number_input("Ocio y Otros", value=200, step=10)

    # Sección 5: Configuración Mercado (Euríbor)
    caminos_eur = []
    n_sims = 1
    
    if modo_h != "FIJA":
        st.markdown("---")
        with st.expander("Previsiones Euríbor", expanded=False):
            modo_prev = st.radio("Modelo de Previsión", ["Estocástico (Monte Carlo)", "Manual (Ajuste fino)"])
            
            if modo_prev == "Estocástico (Monte Carlo)":
                n_sims = st.select_slider("Número de Simulaciones", [10, 50, 100, 250, 500], value=100)
                st.caption("Parámetros Modelo Vasicek")
                theta = st.slider("Media a largo plazo (θ)", 0.0, 5.0, 2.25)
                sigma = st.slider("Volatilidad (σ)", 0.0, 2.0, 0.60)
                kappa = st.slider("Velocidad reversión (κ)", 0.0, 1.0, 0.30)
                r0 = st.number_input("Euríbor actual", value=2.24)
            else:
                n_sims = 1

        n_años_var = anios_p if modo_h == "VARIABLE" else max(0, anios_p - anios_fijos)
        
        if n_años_var > 0:
            if modo_prev == "Manual (Ajuste fino)":
                st.info("Ajuste manual de tipos futuros")
                eur_list = []
                for i in range(n_años_var):
                     eur_list.append(st.slider(f"Año {i + 1 + anios_fijos}", -1.0, 7.0, 3.2, key=f"e{i}", help="Tipo estimado"))
                caminos_eur = [eur_list]
            else:
                caminos_eur = simular_vasicek(r0, theta, kappa, sigma, n_años_var, n_sims)
    else:
        caminos_eur = [[0.0] * anios_p]
        n_sims = 1


# ==========================================
# 3. AREA PRINCIPAL
# ==========================================

st.title("Simulador Financiero Hipotecario")
st.markdown("Análisis detallado de costes, riesgos y proyección patrimonial.")

# --- BARRA DE AMORTIZACIÓN EXTRA ---
with st.expander("Planificación de Amortizaciones Extraordinarias", expanded=False):
    st.info("Define aportaciones de capital anuales para reducir la deuda anticipadamente.")
    cols_a = st.columns(6) 
    amort_list = [cols_a[i%6].number_input(f"Año {i+1}", 0, 50000, 0, step=500, key=f"a{i}") for i in range(anios_p)]

# --- CÁLCULO ---
kpis_int, kpis_ahorro, kpis_pat, kpis_seguros = [], [], [], []
cuotas_matrix, eur_matrix = [], []
df_median, df_base_median = None, None

total_gastos_vida_mensual = g_comida + g_suministros + g_gasolina + g_otros
coste_mensual_seguros = (s_hogar + s_vida) / 12

if n_sims > 50: 
    bar_text = st.empty()
    bar = st.progress(0)

for i, camino in enumerate(caminos_eur):
    # Escenario Actual
    df = calcular_hipoteca_core(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino, amort_list, tipo_reduc, es_autopromotor, meses_carencia)
    # Escenario Base 
    df_base = calcular_hipoteca_core(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino, [0]*anios_p, 'PLAZO', es_autopromotor, meses_carencia)
    
    # Cálculos Patrimoniales
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

if n_sims > 50: 
    bar.empty()
    bar_text.empty()

# Selección de escenario Mediana
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

# Métricas Globales
intereses_totales = np.median(kpis_int)
seguros_totales = np.median(kpis_seguros)
coste_total_operacion = intereses_totales + seguros_totales
ahorro_intereses = np.median(kpis_ahorro)
meses_con_hipoteca = len(df_median[df_median['Saldo'] > 0])
meses_ahorrados = (anios_p * 12) - meses_con_hipoteca

# Cuota primer mes
idx_ref = 0 if not es_autopromotor else meses_carencia
cuota_inicial = df_median.iloc[idx_ref]['Cuota']
tasa_inicial = df_median.iloc[idx_ref]['Tasa']

# ==========================================
# 4. DASHBOARD DE RESULTADOS
# ==========================================

st.divider()

# FILA DE KPIs
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

with col_kpi1:
    st.metric(label="Cuota Mensual Estimada", value=f"{cuota_inicial:,.2f} €", delta=f"Tasa: {tasa_inicial:.2f}%")
with col_kpi2:
    st.metric(label="Total Intereses a Pagar", value=f"{intereses_totales:,.0f} €", delta_color="inverse")
with col_kpi3:
    st.metric(label="Coste Real (Int. + Seguros)", value=f"{coste_total_operacion:,.0f} €", delta_color="inverse")
with col_kpi4:
    if ahorro_intereses > 100:
        st.metric(label="Ahorro por Amortizar", value=f"{ahorro_intereses:,.0f} €", delta=f"-{meses_ahorrados // 12} años")
    else:
        st.metric(label="Capital Pendiente", value=f"{capital_init:,.0f} €")

st.divider()

# PESTAÑAS DE ANÁLISIS
tabs = st.tabs(["📊 Evolución Temporal", "📉 Impacto Amortización", "💰 Patrimonio Neto", "🎲 Análisis de Riesgo"])

# --- TAB 1: TIPOS Y CUOTAS ---
with tabs[0]:
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("Trayectoria Euríbor (Con Incertidumbre)")
        if modo_h == "FIJA":
            st.info("Hipoteca a Tipo Fijo: Sin exposición a fluctuaciones de mercado.")
        else:
            mat_eur = np.array(eur_matrix)
            
            # --- CORRECCIÓN: Cálculo de percentiles para visualizar incertidumbre ---
            p50_eur = np.percentile(mat_eur, 50, axis=0)
            p10_eur = np.percentile(mat_eur, 10, axis=0) # Banda inferior
            p90_eur = np.percentile(mat_eur, 90, axis=0) # Banda superior
            
            anios_x = np.arange(1, len(p50_eur)+1)
            
            fig_e = go.Figure()
            
            # Dibujar bandas de confianza (Fan Chart)
            fig_e.add_trace(go.Scatter(
                x=anios_x, y=p90_eur,
                mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip'
            ))
            fig_e.add_trace(go.Scatter(
                x=anios_x, y=p10_eur,
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(52, 152, 219, 0.2)', # Azul suave transparente
                name='Rango Probable (80%)', showlegend=True
            ))
            
            # Línea Mediana
            fig_e.add_trace(go.Scatter(x=anios_x, y=p50_eur, line=dict(color='#3498db', width=3), name='Euríbor Mediana'))
            fig_e.update_layout(template="plotly_white", height=350, margin=dict(t=20,b=20), xaxis_title="Año", yaxis_title="Tipo %")
            st.plotly_chart(fig_e, use_container_width=True)

    with col_g2:
        st.subheader("Proyección de Cuota")
        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(x=df_median['Mes'], y=df_median['Cuota'], fill='tozeroy', line=dict(color='#2c3e50', width=2), name='Cuota Mensual'))
        if es_autopromotor:
            fig_q.add_vline(x=meses_carencia, line_dash="dot", line_color="green", annotation_text="Fin Carencia")
        
        fig_q.update_layout(template="plotly_white", height=350, margin=dict(t=20,b=20), xaxis_title="Mes", yaxis_title="Euros")
        st.plotly_chart(fig_q, use_container_width=True)

# --- TAB 2: AMORTIZACIÓN ---
with tabs[1]:
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        st.subheader("Intereses Acumulados")
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=df_base_median['Mes'], y=df_base_median['Intereses'].cumsum(), name='Escenario Base', line=dict(color='#95a5a6', dash='dash')))
        fig_r.add_trace(go.Scatter(x=df_median['Mes'], y=df_median['Intereses'].cumsum(), name='Con Amortizaciones', line=dict(color='#e74c3c', width=3)))
        fig_r.update_layout(template="plotly_white", height=350, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_r, use_container_width=True)
    
    with col_a2:
        st.subheader("Velocidad de Pago de Deuda")
        fig_race = go.Figure()
        fig_race.add_trace(go.Scatter(x=df_base_median['Mes'], y=df_base_median['Saldo'], name='Saldo Base', line=dict(color='#95a5a6', dash='dash'))) 
        fig_race.add_trace(go.Scatter(x=df_median['Mes'], y=df_median['Saldo'], fill='tozeroy', name='Saldo Real', line=dict(color='#27ae60')))
        fig_race.update_layout(template="plotly_white", height=350, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_race, use_container_width=True)

# --- TAB 3: PATRIMONIO ---
with tabs[2]:
    st.subheader("Evolución de Riqueza Neta")
    st.caption("Proyección basada en Ingresos - Gastos de Vida - Pagos Hipoteca + Valor Vivienda")
    
    fig_nw = go.Figure()
    
    gasto_base = df_base_median['Cuota'] + coste_mensual_seguros + total_gastos_vida_mensual
    ahorro_base = ahorro_inicial + (ingresos - gasto_base).cumsum()
    pat_base = ahorro_base + (precio_vivienda - df_base_median['Saldo'])
    
    fig_nw.add_trace(go.Scatter(x=df_median['Mes'], y=df_median['Patrimonio'], name='Patrimonio (Estrategia Actual)', line=dict(color='#8e44ad', width=4)))
    fig_nw.add_trace(go.Scatter(x=df_base_median['Mes'], y=pat_base, name='Patrimonio (Sin Amortizar)', line=dict(color='#bdc3c7', dash='dot')))
    
    fig_nw.update_layout(template="plotly_white", height=400, hovermode="x unified", legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig_nw, use_container_width=True)

# --- TAB 4: RIESGO ---
with tabs[3]:
    if n_sims > 1 and modo_h != "FIJA":
        p5_int = np.percentile(kpis_int, 5)
        p95_int = np.percentile(kpis_int, 95)
        p10_int = np.percentile(kpis_int, 10)
        p90_int = np.percentile(kpis_int, 90)
        
        st.subheader("Análisis de Sensibilidad (Monte Carlo)")
        st.markdown("Basado en la volatilidad configurada, estos son los rangos probables de intereses totales a pagar:")
        
        c_risk1, c_risk2 = st.columns(2)
        with c_risk1:
            st.warning(f"Escenario Adverso (90%): Pagarías hasta **{p90_int:,.0f} €** de intereses.")
        with c_risk2:
            st.success(f"Escenario Favorable (10%): Pagarías solo **{p10_int:,.0f} €** de intereses.")
            
        st.progress(0.9, text=f"Rango de máxima probabilidad (5%-95%): {p5_int:,.0f} € - {p95_int:,.0f} €")
        
    elif modo_h == "FIJA":
        st.success("Operación sin riesgo de tipo de interés. Coste fijo garantizado.")
    else:
        st.info("Para ver el análisis de riesgo, selecciona el método 'Estocástico' en la configuración del Euríbor.")

# ==========================================
# 5. TABLA DE DATOS
# ==========================================
with st.expander("Ver Tabla de Amortización Detallada"):
    numeric_cols = df_median.select_dtypes(include=[np.number]).columns
    st.dataframe(
        df_median.style.format("{:.2f}", subset=numeric_cols),
        use_container_width=True,
        height=400
    )
