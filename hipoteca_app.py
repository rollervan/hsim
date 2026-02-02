import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="Simulador Hipotecario & Patrimonial 3.0",
    page_icon="🏦",
    layout="wide"
)

# ==========================================
# 1. MOTOR MATEMÁTICO (CORE)
# ==========================================
def calcular_hipoteca_core(capital, anios, diferencial, tipo_fijo, anios_fijos, modo, euribor_puntos, amortizaciones, tipo_reduc):
    """
    Calcula la tabla de amortización mes a mes con lógica de reducción (Cuota vs Plazo)
    y redondeo financiero a 2 decimales.
    """
    n_meses_total = anios * 12
    saldo_real = round(float(capital), 2)
    saldo_teorico = round(float(capital), 2) # Sombra para mantener cuota en reducción de plazo
    data = []
    mes_global = 1
    
    # Relleno de seguridad para las listas
    puntos_eur = list(euribor_puntos) + [euribor_puntos[-1]] * (max(0, anios - len(euribor_puntos)))
    puntos_amort = list(amortizaciones) + [0] * (max(0, anios - len(amortizaciones)))

    idx_var = 0 
    for anio in range(anios):
        if saldo_real <= 0: break # Hipoteca pagada

        # 1. Determinar Tipo de Interés del Año
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
        
        # Lógica Plazo vs Cuota
        base_calc = saldo_teorico if tipo_reduc == 'PLAZO' else saldo_real
        if base_calc < saldo_real: base_calc = saldo_real

        # Cálculo de Cuota (Sistema Francés)
        if base_calc <= 0.01: 
            cuota = 0
        else:
            if tasa_mensual > 0:
                cuota = base_calc * (tasa_mensual * (1 + tasa_mensual)**meses_restantes) / ((1 + tasa_mensual)**meses_restantes - 1)
            else: 
                cuota = base_calc / meses_restantes
        
        cuota = round(cuota, 2)

        # 2. Bucle Mensual
        for m in range(12):
            if saldo_real <= 0.009: 
                saldo_real = 0
                break

            interes_m = round(saldo_real * tasa_mensual, 2)
            capital_m = round(cuota - interes_m, 2)
            
            # Ajuste final de deuda
            if capital_m > saldo_real:
                capital_m = saldo_real
                cuota = round(capital_m + interes_m, 2)

            saldo_real = round(saldo_real - capital_m, 2)
            
            # Ajuste del saldo teórico (Sombra)
            int_teorico = round(saldo_teorico * tasa_mensual, 2)
            amort_teorica = round(cuota - int_teorico, 2)
            saldo_teorico = round(saldo_teorico - amort_teorica, 2)
            if saldo_teorico < 0: saldo_teorico = 0

            data.append({
                'Mes': mes_global, 
                'Año': anio + 1, 
                'Tasa': tasa_anual, 
                'Cuota': cuota, 
                'Intereses': interes_m, 
                'Capital': capital_m, 
                'Saldo': saldo_real, 
                'Amort_Extra': 0
            })
            
            # 3. Amortización Extra (Mes 12)
            if m == 11 and saldo_real > 0 and puntos_amort[anio] > 0:
                ejec = round(min(puntos_amort[anio], saldo_real), 2)
                saldo_real = round(saldo_real - ejec, 2)
                
                if tipo_reduc == 'CUOTA': 
                    saldo_teorico = saldo_real # Recalcula cuota a la baja
                # Si es PLAZO, no tocamos saldo_teorico, manteniendo la cuota alta

                data[-1]['Amort_Extra'] = ejec
                data[-1]['Capital'] = round(data[-1]['Capital'] + ejec, 2)
            
            mes_global += 1

    return pd.DataFrame(data)

def simular_vasicek(r0, theta, kappa, sigma, anios, n_sims=100):
    """Genera caminos estocásticos de tipos de interés (Modelo Vasicek)"""
    dt = 1 
    sims = []
    for _ in range(n_sims):
        camino = [r0]
        for t in range(anios - 1):
            dr = kappa * (theta - camino[-1]) * dt + sigma * np.random.normal()
            nuevo_r = camino[-1] + dr
            camino.append(max(-1.0, nuevo_r)) # Suelo en -1%
        sims.append(camino)
    return np.array(sims)

# ==========================================
# 2. INTERFAZ DE USUARIO (SIDEBAR)
# ==========================================
st.title("🏦 Simulador Hipotecario & Patrimonial 3.0")
st.markdown("---")

with st.sidebar:
    st.header("👤 Perfil Financiero")
    ingresos = st.number_input("Ingresos Netos Mensuales (€)", value=3000, step=100)
    ahorro_inicial = st.number_input("Ahorro Líquido Inicial (€)", value=20000, step=1000)
    precio_vivienda = st.number_input("Valor de la Vivienda (€)", value=220000, step=5000)
    
    st.markdown("---")
    st.header("⚙️ Préstamo")
    modo_h = st.selectbox("Modalidad", ["MIXTA", "VARIABLE", "FIJA"])
    tipo_reduc = st.radio("Estrategia Amortización", ["PLAZO", "CUOTA"], help="PLAZO: Acabas antes manteniendo cuota. CUOTA: Bajas mensualidad.")
    capital_init = st.number_input("Capital Pendiente (€)", value=180000, step=1000)
    anios_p = st.number_input("Años Restantes", value=25, min_value=1)
    
    st.markdown("---")
    st.header("🛡️ Costes Ocultos")
    s_hogar = st.number_input("Seguro Hogar (Anual €)", value=300)
    s_vida = st.number_input("Seguro Vida (Anual €)", value=400)
    gastos_fijos = st.number_input("Comunidad/IBI (Mensual €)", value=100)

    st.markdown("---")
    st.header("📈 Previsión Euríbor")
    modo_prev = st.radio("Método", ["Estocástico (Monte Carlo)", "Manual (Sliders)"])
    
    if modo_prev == "Estocástico (Monte Carlo)":
        n_sims = st.select_slider("Nº Simulaciones", options=[10, 50, 100, 500], value=100)
        st.caption("Parámetros Vasicek (Consenso 2026)")
        theta = st.slider("Media L/P (θ)", 0.0, 5.0, 2.25, 0.1)
        sigma = st.slider("Volatilidad (σ)", 0.0, 2.0, 0.60, 0.1)
        kappa = st.slider("Inercia (κ)", 0.0, 1.0, 0.30, 0.1)
        r0 = st.number_input("Euríbor Actual %", value=2.24)
    else:
        n_sims = 1

    st.markdown("---")
    st.header("🏦 Condiciones Banco")
    tipo_fijo = st.number_input("Tipo Fijo (%)", value=2.25)
    anios_fijos = st.number_input("Años fijo", value=5) if modo_h == "MIXTA" else 0
    diferencial = st.number_input("Diferencial %", value=0.55)

# ==========================================
# 3. CONFIGURACIÓN DINÁMICA (MAIN PAGE)
# ==========================================

# 1. Generación de Euríbor
n_años_var = anios_p if modo_h == "VARIABLE" else max(0, anios_p - anios_fijos)
if modo_prev == "Manual (Sliders)":
    c1_e, c2_e = st.columns([1, 3])
    with c1_e: st.info("Configura la curva del Euríbor manualmente:")
    eur_list = []
    with st.expander("🛠️ Ajustar Euríbor Año a Año", expanded=True):
        cols = st.columns(4)
        for i in range(n_años_var):
            with cols[i % 4]:
                eur_list.append(st.slider(f"A{anios_p-n_años_var+i+1}", -1.0, 7.0, 2.25, key=f"e{i}"))
    caminos_eur = [eur_list]
else:
    caminos_eur = simular_vasicek(r0, theta, kappa, sigma, n_años_var, n_sims)

# 2. Configuración Amortizaciones
st.subheader("💰 Plan de Amortización Anticipada")
amort_list = []
with st.expander("Configurar Aportaciones Extra Anuales", expanded=False):
    cols_a = st.columns(4)
    for i in range(anios_p):
        with cols_a[i % 4]:
            amort_list.append(st.slider(f"Año {i+1}", 0, 20000, 0, step=500, key=f"a{i}"))

# ==========================================
# 4. CÁLCULO Y PROCESAMIENTO
# ==========================================

# Listas para guardar resultados de las N simulaciones
sim_resultados = [] # Guardará DataFrames (solo necesitamos el central para tabla)
kpis_intereses = []
kpis_ahorro_amort = []
kpis_equity_final = []
cuotas_matrix = [] # Para el gráfico de abanico (Fan Chart)

df_median = None # Dataframe del escenario central

# Barra de progreso si hay muchas simulaciones
if n_sims > 50:
    progress_bar = st.progress(0)

for i, camino in enumerate(caminos_eur):
    # Escenario A: Con Amortizaciones (Real)
    df = calcular_hipoteca_core(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino, amort_list, tipo_reduc)
    
    # Escenario B: Sin Amortizaciones (Base para comparar ahorro)
    df_base = calcular_hipoteca_core(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino, [0]*anios_p, 'PLAZO')
    
    # --- CÁLCULOS PATRIMONIALES ---
    # Gastos Mensuales Totales
    gasto_mensual_extra = (s_hogar + s_vida) / 12 + gastos_fijos
    df['Gasto_Total'] = df['Cuota'] + gasto_mensual_extra
    
    # Capacidad de Ahorro
    df['Ahorro_Mensual_Flow'] = ingresos - df['Gasto_Total']
    
    # Evolución Ahorro Líquido (Acumulado + Nuevo Ahorro - Amortizaciones Extra)
    # Nota: Restamos Amort_Extra porque sale del bolsillo del ahorro
    df['Ahorro_Liquido'] = ahorro_inicial + df['Ahorro_Mensual_Flow'].cumsum() - df['Amort_Extra'].cumsum()
    
    # Equity (Valor de la casa que es tuyo = Precio - Deuda)
    df['Equity'] = precio_vivienda - df['Saldo']
    
    # Patrimonio Neto (Net Worth)
    df['Patrimonio_Neto'] = df['Ahorro_Liquido'] + df['Equity']

    # Guardar métricas
    intereses_totales = df['Intereses'].sum()
    ahorro_generado = df_base['Intereses'].sum() - intereses_totales
    
    kpis_intereses.append(intereses_totales)
    kpis_ahorro_amort.append(ahorro_generado)
    kpis_equity_final.append(df['Patrimonio_Neto'].iloc[-1])
    cuotas_matrix.append(df['Cuota'].values)
    
    # Guardamos el primer DF como "central" temporalmente, luego buscamos la mediana
    if i == 0: df_median = df
    
    if n_sims > 50: progress_bar.progress((i+1)/n_sims)

# Encontrar el índice de la mediana para mostrar el DF más representativo
idx_median = np.argsort(kpis_intereses)[len(kpis_intereses)//2]
# Recalculamos el DF de la mediana para asegurar que coincide con los gráficos
camino_mediana = caminos_eur[idx_median] if n_sims > 1 else caminos_eur[0]
df_median = calcular_hipoteca_core(capital_init, anios_p, diferencial, tipo_fijo, anios_fijos, modo_h, camino_mediana, amort_list, tipo_reduc)
# Reaplicar lógica patrimonial al DF Mediana (para gráficos detallados)
gasto_mensual_extra = (s_hogar + s_vida) / 12 + gastos_fijos
df_median['Gasto_Total'] = df_median['Cuota'] + gasto_mensual_extra
df_median['Ahorro_Mensual_Flow'] = ingresos - df_median['Gasto_Total']
df_median['Ahorro_Liquido'] = ahorro_inicial + df_median['Ahorro_Mensual_Flow'].cumsum() - df_median['Amort_Extra'].cumsum()
df_median['Equity'] = precio_vivienda - df_median['Saldo']
df_median['Patrimonio_Neto'] = df_median['Ahorro_Liquido'] + df_median['Equity']

# ==========================================
# 5. DASHBOARD DE RESULTADOS
# ==========================================
col1, col2, col3, col4 = st.columns(4)

med_int = np.median(kpis_intereses)
med_ahorro = np.median(kpis_ahorro_amort)
med_patrimonio = np.median(kpis_equity_final)
tasa_esfuerzo = (df_median['Gasto_Total'].mean() / ingresos) * 100

col1.metric("Intereses Totales", f"{med_int:,.0f} €", help="Mediana de todas las simulaciones")
col2.metric("Ahorro Intereses", f"{med_ahorro:,.0f} €", delta="Generado por amortizar", delta_color="normal")
col3.metric("Patrimonio Final", f"{med_patrimonio:,.0f} €", help="Ahorros + Valor Casa al final")
col4.metric("Tasa Esfuerzo Media", f"{tasa_esfuerzo:.1f} %", delta="- Ideal < 30%", delta_color="inverse")

if n_sims > 1:
    p95_int = np.percentile(kpis_intereses, 95)
    st.warning(f"⚠️ **Análisis de Riesgo (P95):** Hay un 5% de probabilidad de que los intereses suban hasta **{p95_int:,.0f} €**.")

st.markdown("---")

# --- GRÁFICOS (PLOTLY) ---
tab1, tab2, tab3 = st.tabs(["📊 Cuota & Riesgo", "📈 Evolución Patrimonio", "💸 Desglose Gastos"])

with tab1:
    col_g1, col_g2 = st.columns(2)
    
    # 1. Gráfico Fan Chart de Cuotas
    with col_g1:
        st.subheader("Evolución Cuota Mensual")
        # Preprocesar matriz de cuotas (rellenar con NaN para longitudes distintas si se paga antes)
        max_len = max(len(c) for c in cuotas_matrix)
        matrix_padded = np.array([np.pad(c, (0, max_len - len(c)), constant_values=np.nan) for c in cuotas_matrix])
        
        meses = np.arange(1, max_len + 1)
        p5 = np.nanpercentile(matrix_padded, 5, axis=0)
        p50 = np.nanpercentile(matrix_padded, 50, axis=0)
        p95 = np.nanpercentile(matrix_padded, 95, axis=0)
        
        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(x=np.concatenate([meses, meses[::-1]]), y=np.concatenate([p95, p5[::-1]]),
                                   fill='toself', fillcolor='rgba(231, 76, 60, 0.2)', line=dict(color='rgba(255,255,255,0)'),
                                   name='Incertidumbre (P5-P95)', hoverinfo="skip"))
        fig_q.add_trace(go.Scatter(x=meses, y=p50, line=dict(color='#c0392b', width=2), name='Cuota Mediana',
                                   hovertemplate='Mes %{x}: <b>%{y:.0f}€</b>'))
        fig_q.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_q, use_container_width=True)

    # 2. Gráfico Interés vs Capital (Escenario Central)
    with col_g2:
        st.subheader("Carrera: Interés vs Capital")
        int_restante = df_median['Intereses'].sum() - df_median['Intereses'].cumsum()
        cap_amortizado = df_median['Capital'].cumsum()
        
        fig_race = go.Figure()
        fig_race.add_trace(go.Scatter(x=df_median['Mes'], y=cap_amortizado, fill='tozeroy', name='Capital Pagado', line=dict(color='#27ae60')))
        fig_race.add_trace(go.Scatter(x=df_median['Mes'], y=int_restante, fill='tozeroy', name='Interés Pendiente', line=dict(color='#e67e22')))
        fig_race.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_race, use_container_width=True)

with tab2:
    st.subheader("Tu Riqueza Neta (Net Worth)")
    fig_nw = go.Figure()
    fig_nw.add_trace(go.Scatter(x=df_median['Mes'], y=df_median['Patrimonio_Neto'], name='Patrimonio Total', line=dict(color='#8e44ad', width=4)))
    fig_nw.add_trace(go.Scatter(x=df_median['Mes'], y=df_median['Equity'], name='Valor Propiedad (Equity)', stackgroup='one', line=dict(color='#2980b9')))
    fig_nw.add_trace(go.Scatter(x=df_median['Mes'], y=df_median['Ahorro_Liquido'], name='Ahorro Líquido', stackgroup='one', line=dict(color='#2ecc71')))
    fig_nw.update_layout(height=400, hovermode="x unified")
    st.plotly_chart(fig_nw, use_container_width=True)

with tab3:
    st.subheader("¿A dónde va mi sueldo?")
    # Agrupar por años para que el gráfico de barras no sea ilegible
    df_anual = df_median.groupby('Año')[['Cuota']].sum()
    df_anual['Seguros'] = s_hogar + s_vida
    df_anual['Otros'] = gastos_fijos * 12
    
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=df_anual.index, y=df_anual['Cuota'], name='Hipoteca', marker_color='#3498db'))
    fig_bar.add_trace(go.Bar(x=df_anual.index, y=df_anual['Seguros'], name='Seguros', marker_color='#f1c40f'))
    fig_bar.add_trace(go.Bar(x=df_anual.index, y=df_anual['Otros'], name='Comunidad/IBI', marker_color='#95a5a6'))
    fig_bar.update_layout(barmode='stack', height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# 6. EXPORTACIÓN DATOS
# ==========================================
st.markdown("---")
with st.expander("📥 Descargar Datos Detallados (Excel/CSV)"):
    st.dataframe(df_median.style.format("{:.2f}"))
    csv = df_median.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar Cuadro de Amortización Completo",
        data=csv,
        file_name='simulacion_hipoteca_pro.csv',
        mime='text/csv'
    )
