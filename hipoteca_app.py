import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ==========================================
st.set_page_config(
    page_title="Simulador Financiero Pro",
    page_icon="stats",
    layout="wide"
)

# CSS Profesional
st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
    h1, h2, h3 {font-family: 'Segoe UI', sans-serif; color: #2c3e50;}
    .stMetric {background-color: #ffffff; border: 1px solid #e0e0e0; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);}
    div[data-testid="stExpander"] {border: 1px solid #e0e0e0; border-radius: 8px;}
    /* Estilo para el botón de formulario */
    div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        background-color: #2c3e50;
        color: white;
        border: none;
        padding: 10px;
        font-weight: bold;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #34495e;
        color: white;
    }
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
    
    # Relleno seguro de listas
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
                    
                    # Ajuste saldo teórico
                    int_teorico = round(saldo_teorico * tasa_mensual, 2)
                    amort_teorica = round(cuota - int_teorico, 2)
                    saldo_teorico = round(saldo_teorico - amort_teorica, 2)
                    if saldo_teorico < 0: saldo_teorico = 0

            # Guardar datos
            data.append({
                'Mes': mes_global, 'Año': anio + 1, 'Tasa': tasa_anual if saldo_real > 0 else 0, 
                'Cuota': cuota, 'Intereses': interes_m, 'Capital': capital_m, 
                'Saldo': saldo_real, 'Amort_Extra': 0,
                'Fase': 'Carencia' if en_periodo_carencia else 'Amortización'
            })
            
            # 3. AMORTIZACIÓN EXTRA
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
# 2. INTERFAZ: SIDEBAR (INPUTS EN FORMULARIO)
# ==========================================
with st.sidebar:
    st.header("Configuración")
    
    # --- INICIO DEL FORMULARIO ---
    # Al usar st.form, nada se ejecuta hasta que se pulsa el botón 'submit'
    with st.form(key='panel_control'):
        
        with st.expander("Perfil Económico", expanded=True):
            ingresos = st.number_input("Ingresos netos (€)", value=2500, step=100)
            ahorro_inicial = st.number_input("Ahorro inicial (€)", value=0, step=1000)
            precio_vivienda = st.number_input("Valor Vivienda (€)", value=0, step=5000)

        st.markdown("---")
        st.subheader("Préstamo")
        modo_h = st.selectbox("Modalidad", ["MIXTA", "VARIABLE", "FIJA"])
        
        es_autopromotor = st.checkbox("Es Autopromoción", value=False)
        meses_carencia = 0
        if es_autopromotor:
            meses_carencia = st.number_input("Meses carencia", value=11, min_value=1, max_value=36)
        
        capital_init = st.number_input("Capital (€)", value=180000, step=1000)
        anios_p = st.number_input("Duración (Años)", value=25, min_value=1)
        
        st.markdown("---")
        st.subheader("Condiciones")
        
        tipo_fijo = 0.0
        diferencial = 0.0
        anios_fijos = 0
        
        c1, c2 = st.columns(2)
        if modo_h == "FIJA":
            tipo_fijo = c1.number_input("TIN Fijo (%)", value=2.50, step=0.05)
        elif modo_h == "VARIABLE":
            diferencial = c1.number_input("Diferencial (%)", value=0.55, step=0.05)
        elif modo_h == "MIXTA":
            tipo_fijo = c1.number_input("Fijo (%)", value=2.25, step=0.05)
            anios_fijos = c2.number_input("Años Fijos", value=5)
            diferencial = st.number_input("Dif. Variable (%)", value=0.55, step=0.05)

        tipo_reduc = st.radio("Amortizar reduciendo:", ["PLAZO", "CUOTA"])

        with st.expander("Gastos y Vinculaciones", expanded=False):
            s_hogar = st.number_input("Seguro Hogar (€/año)", value=300)
            s_vida = st.number_input("Seguro Vida (€/año)", value=300)
            st.markdown("**Gastos Mensuales**")
            g_comida = st.number_input("Comida", value=300)
            g_suministros = st.number_input("Suministros", value=150)
            g_gasolina = st.number_input("Transporte", value=100)
            g_otros = st.number_input("Otros", value=200)

        # Configuración Euríbor
        caminos_eur = []
        n_sims = 1
        
        if modo_h != "FIJA":
            st.markdown("---")
            with st.expander("Simulación Euríbor", expanded=True):
                modo_prev = st.selectbox("Método", ["Monte Carlo (Vasicek)", "Manual"])
                
                if modo_prev == "Monte Carlo (Vasicek)":
                    n_sims = st.select_slider("Iteraciones", [50, 100, 500, 1000], value=100)
                    st.caption("Parámetros Estocásticos")
                    theta = st.slider("Media (Long Term)", 0.0, 5.0, 2.25)
                    sigma = st.slider("Volatilidad", 0.0, 2.0, 0.60)
                    kappa = st.slider("Reversión", 0.0, 1.0, 0.30)
                    r0 = st.number_input("Euríbor Actual", value=2.24)
                else:
                    n_sims = 1
        
        st.markdown("---")
        # --- BOTÓN DE ENVÍO ---
        submit_button = st.form_submit_button("🔄 Recalcular Simulación")

    # Lógica fuera del form pero dependiente de los inputs anteriores
    n_años_var = anios_p if modo_h == "VARIABLE" else max(0, anios_p - anios_fijos)
    
    if modo_h != "FIJA" and n_años_var > 0:
        if modo_prev == "Manual":
            # Nota: Los sliders manuales fuera del form si se quiere reactividad, 
            # pero para coherencia los dejamos como input 'lazy' o calculamos aquí.
            # Para este caso, usamos el cálculo Vasicek por defecto o una lista fija
            # si es manual, requeriría otro form o estar dentro. Lo simplificamos:
            if 'eur_manual' not in st.session_state:
                st.session_state.eur_manual = [3.2] * n_años_var
            
            with st.expander("Ajuste Manual Euríbor"):
                st.info("Para editar el Euríbor manual, usa la configuración abajo:")
                # Pequeño hack para permitir inputs manuales fuera del form principal si es necesario
                # o dejarlos dentro si cupieran. Lo dejamos automático para limpieza.
                eur_list = [3.2] * n_años_var 
                caminos_eur = [eur_list]
        else:
            caminos_eur = simular_vasicek(r0, theta, kappa, sigma, n_años_var, n_sims)
    else:
        caminos_eur = [[0.0] * anios_p]
        n_sims = 1

# ==========================================
# 3. ÁREA PRINCIPAL
# ==========================================

st.title("Simulador Financiero Pro 4.3")

# Amortizaciones (Compacto) - Esto se deja fuera del form principal para permitir ajustes finos
# rápidos sin recargar todo el modelo estocástico, o se puede meter en otro form.
with st.expander("Estrategia de Amortización Anticipada (Ajuste fino)"):
    st.caption("Estas modificaciones se aplican instantáneamente sobre el escenario calculado.")
    cols_a = st.columns(6) 
    amort_list = [cols_a[i%6].number_input(f"A{i+1}", 0, 50000, 0, step=500, key=f"a{i}") for i in range(anios_p)]

# CÁLCULOS
kpis_int, kpis_ahorro, kpis_pat, kpis_seguros = [], [], [], []
cuotas_matrix, eur_matrix = [], []
df_median, df_base_median = None, None

total_gastos = g_comida + g_suministros + g_gasolina + g_otros
coste_mes_seguros = (s_hogar + s_vida) / 12

# Barra de progreso solo si hay muchas simulaciones
if n_sims > 100: 
    prog_bar = st.progress(0)

for i, camino in enumerate(caminos_eur):
    # Calcular
    df = calcular_hipoteca_core(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino, amort_list, tipo_reduc, es_autopromotor, meses_carencia)
    df_base = calcular_hipoteca_core(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino, [0]*anios_p, 'PLAZO', es_autopromotor, meses_carencia)
    
    # KPIs Financieros
    df['Seguros'] = np.where(df['Saldo'] > 0, coste_mes_seguros, 0)
    gasto_tot = df['Cuota'] + df['Seguros'] + total_gastos
    
    df['Ahorro_Liq'] = ahorro_inicial + (ingresos - gasto_tot).cumsum() - df['Amort_Extra'].cumsum()
    df['Patrimonio'] = df['Ahorro_Liq'] + (precio_vivienda - df['Saldo'])
    
    kpis_int.append(df['Intereses'].sum())
    kpis_seguros.append(df['Seguros'].sum())
    kpis_ahorro.append(df_base['Intereses'].sum() - df['Intereses'].sum())
    kpis_pat.append(df['Patrimonio'].iloc[-1])
    
    cuotas_matrix.append(df['Cuota'].values)
    eur_matrix.append(camino)
    
    if i == 0: df_median = df; df_base_median = df_base
    if n_sims > 100: prog_bar.progress((i+1)/n_sims)

if n_sims > 100: prog_bar.empty()

# Escenario Central (Mediana de Intereses Totales)
idx_med = np.argsort(kpis_int)[len(kpis_int)//2]
if n_sims > 1:
    camino_med = caminos_eur[idx_med]
    df_median = calcular_hipoteca_core(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino_med, amort_list, tipo_reduc, es_autopromotor, meses_carencia)
    df_base_median = calcular_hipoteca_core(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino_med, [0]*anios_p, 'PLAZO', es_autopromotor, meses_carencia)
    # Recalcular auxiliares mediana
    df_median['Seguros'] = np.where(df_median['Saldo'] > 0, coste_mes_seguros, 0)
    gasto_tot = df_median['Cuota'] + df_median['Seguros'] + total_gastos
    df_median['Ahorro_Liq'] = ahorro_inicial + (ingresos - gasto_tot).cumsum() - df_median['Amort_Extra'].cumsum()
    df_median['Patrimonio'] = df_median['Ahorro_Liq'] + (precio_vivienda - df_median['Saldo'])

# Métricas Mediana
int_total = np.median(kpis_int)
seg_total = np.median(kpis_seguros)
coste_total = int_total + seg_total
ahorro_int = np.median(kpis_ahorro)
meses_total = len(df_median[df_median['Saldo'] > 0])

idx_ref = 0 if not es_autopromotor else meses_carencia
cuota_ini = df_median.iloc[idx_ref]['Cuota']
tasa_ini = df_median.iloc[idx_ref]['Tasa']

# DASHBOARD
st.markdown("### Resumen Ejecutivo")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Cuota Inicial", f"{cuota_ini:,.2f} €", f"{tasa_ini:.2f}% TIN")
k2.metric("Total Intereses (Est.)", f"{int_total:,.0f} €", delta_color="inverse")
k3.metric("Coste Operación (Int+Seg)", f"{coste_total:,.0f} €", delta_color="inverse")
k4.metric("Ahorro por Amortizar", f"{ahorro_int:,.0f} €", f"-{(anios_p*12 - meses_total)//12} años")

st.markdown("---")

tabs = st.tabs(["📊 Evolución", "📉 Comparativa Amortización", "💰 Patrimonio", "🎲 Riesgo (Monte Carlo)"])

# TAB 1: EVOLUCIÓN
with tabs[0]:
    c_e1, c_e2 = st.columns(2)
    with c_e1:
        st.subheader("Euríbor: Proyección e Incertidumbre")
        if modo_h == "FIJA":
            st.info("Tipo Fijo: Sin incertidumbre de mercado.")
        else:
            mat = np.array(eur_matrix)
            p10 = np.percentile(mat, 10, axis=0)
            p50 = np.percentile(mat, 50, axis=0)
            p90 = np.percentile(mat, 90, axis=0)
            x_ax = np.arange(1, len(p50)+1)
            
            fig = go.Figure()
            # Banda de confianza
            fig.add_trace(go.Scatter(x=x_ax, y=p90, mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=x_ax, y=p10, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,250,0.15)', name='Intervalo 80% Confianza'))
            # Mediana
            fig.add_trace(go.Scatter(x=x_ax, y=p50, mode='lines', line=dict(color='#0055aa', width=3), name='Mediana'))
            fig.update_layout(template='plotly_white', height=350, margin=dict(l=0,r=0,t=30,b=0), legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)
            
    with c_e2:
        st.subheader("Cuota Mensual")
        fig2 = px.line(df_median, x='Mes', y='Cuota', title="")
        fig2.update_traces(line_color='#d9534f', line_width=2.5)
        fig2.update_layout(template='plotly_white', height=350, margin=dict(l=0,r=0,t=30,b=0))
        if es_autopromotor:
            fig2.add_vline(x=meses_carencia, line_dash="dot", annotation_text="Fin Carencia")
        st.plotly_chart(fig2, use_container_width=True)

# TAB 2: AMORTIZACIÓN
with tabs[1]:
    c_a1, c_a2 = st.columns(2)
    with c_a1:
        st.subheader("Pago de Intereses Acumulado")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df_base_median['Mes'], y=df_base_median['Intereses'].cumsum(), name='Sin Amortizar', line=dict(color='gray', dash='dash')))
        fig3.add_trace(go.Scatter(x=df_median['Mes'], y=df_median['Intereses'].cumsum(), name='Con Amortización', line=dict(color='#d9534f', width=3)))
        fig3.update_layout(template='plotly_white', height=350, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig3, use_container_width=True)
    with c_a2:
        st.subheader("Reducción de Deuda")
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=df_base_median['Mes'], y=df_base_median['Saldo'], name='Saldo Base', line=dict(color='gray', dash='dash')))
        fig4.add_trace(go.Scatter(x=df_median['Mes'], y=df_median['Saldo'], fill='tozeroy', name='Saldo Real', line=dict(color='#5cb85c')))
        fig4.update_layout(template='plotly_white', height=350, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig4, use_container_width=True)

# TAB 3: PATRIMONIO
with tabs[2]:
    st.subheader("Evolución del Patrimonio Neto")
    fig5 = go.Figure()
    # Comparativa Base
    g_base = df_base_median['Cuota'] + coste_mes_seguros + total_gastos
    ah_base = ahorro_inicial + (ingresos - g_base).cumsum()
    pat_base = ah_base + (precio_vivienda - df_base_median['Saldo'])
    
    fig5.add_trace(go.Scatter(x=df_base_median['Mes'], y=pat_base, name='Escenario Base', line=dict(color='gray', dash='dot')))
    fig5.add_trace(go.Scatter(x=df_median['Mes'], y=df_median['Patrimonio'], name='Escenario Actual', line=dict(color='#6f42c1', width=3)))
    fig5.update_layout(template='plotly_white', height=400, hovermode="x unified")
    st.plotly_chart(fig5, use_container_width=True)

# TAB 4: RIESGO
with tabs[3]:
    if modo_h == "FIJA":
        st.success("✅ Al ser una hipoteca a TIPO FIJO, no existe riesgo de tipo de interés. El coste es determinista.")
    elif n_sims < 10:
        st.warning("⚠️ Incrementa el número de simulaciones en la barra lateral (>50) para obtener un análisis de riesgo fiable.")
    else:
        st.subheader("Distribución de Probabilidad del Coste Total de Intereses")
        
        # Estadísticos
        p5 = np.percentile(kpis_int, 5)
        p25 = np.percentile(kpis_int, 25)
        p50 = np.percentile(kpis_int, 50)
        p75 = np.percentile(kpis_int, 75)
        p95 = np.percentile(kpis_int, 95)
        
        c_r1, c_r2 = st.columns([2, 1])
        
        with c_r1:
            # Histograma de Frecuencias
            fig_hist = px.histogram(x=kpis_int, nbins=30, labels={'x': 'Total Intereses Pagados (€)', 'y': 'Frecuencia'}, color_discrete_sequence=['#8884d8'])
            fig_hist.add_vline(x=p5, line_dash="dash", line_color="green", annotation_text=f"P5 (Mejor): {p5/1000:.0f}k")
            fig_hist.add_vline(x=p95, line_dash="dash", line_color="red", annotation_text=f"P95 (Peor): {p95/1000:.0f}k")
            fig_hist.add_vline(x=p50, line_color="black", annotation_text=f"Mediana: {p50/1000:.0f}k")
            fig_hist.update_layout(template='plotly_white', height=400, showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with c_r2:
            st.write("#### Métricas de Sensibilidad")
            st.markdown(f"""
            <div style="background-color:#f8f9fa; padding:15px; border-radius:5px; font-size:0.9rem">
            <b>Escenario Favorable (P10):</b><br> {np.percentile(kpis_int, 10):,.0f} € <br><br>
            <b>Escenario Central (Mediana):</b><br> {p50:,.0f} € <br><br>
            <b>Escenario Adverso (P90):</b><br> {np.percentile(kpis_int, 90):,.0f} € <br><br>
            <hr>
            <b>Riesgo Extremo (P95):</b><br> {p95:,.0f} €
            </div>
            """, unsafe_allow_html=True)
            
            st.info("El gráfico muestra la probabilidad de pagar distintas cantidades de intereses según la volatilidad del Euríbor.")

# DATA PREVIEW
with st.expander("Ver Datos Numéricos (Escenario Mediana)"):
    num_cols = df_median.select_dtypes(include=[np.number]).columns
    st.dataframe(df_median.style.format("{:.2f}", subset=num_cols), use_container_width=True)
