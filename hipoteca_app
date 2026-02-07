import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Simulador Hipotecario Pro", layout="wide")

# --- ESTILOS CSS ---
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .metric-container { background-color: #f0f2f6; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCIONES DE CÁLCULO ---

def calcular_cuota_mensual(capital, tasa_anual, meses_restantes):
    """Calcula la cuota mensual usando el sistema de amortización francés."""
    if tasa_anual == 0:
        return capital / meses_restantes
    
    i = tasa_anual / 12 / 100
    return capital * (i * (1 + i)**meses_restantes) / ((1 + i)**meses_restantes - 1)

def generar_cuadro_amortizacion(capital, plazo_anos, tipo_hipoteca, params):
    """
    Genera el cuadro de amortización mes a mes.
    Simula revisión de tipo de interés ANUAL para la parte variable.
    """
    plazo_meses = plazo_anos * 12
    saldo_pendiente = capital
    calendario = []
    
    euribor_base = params.get('euribor_proyeccion', 2.5)
    
    # Parámetros específicos
    periodo_fijo_anos = params.get('periodo_fijo', 0)
    tipo_fijo = params.get('tipo_fijo', 0)
    diferencial = params.get('diferencial', 0)
    
    cuota_actual = 0
    tasa_actual = 0
    
    for mes in range(1, plazo_meses + 1):
        ano_actual = (mes - 1) // 12
        
        # Determinar Tasa de Interés del periodo
        if tipo_hipoteca == "Mixta" and mes <= (periodo_fijo_anos * 12):
            tasa_aplicable = tipo_fijo
            modo = "Fijo"
        else:
            # Parte Variable (o toda la hipoteca si es Variable)
            # Simulamos que el Euribor varía ligeramente según la proyección
            # Aquí se podría hacer más complejo con una curva de tipos
            tasa_aplicable = euribor_base + diferencial
            modo = "Variable"

        # Recalcular cuota si cambia el tipo o es revisión anual (mes 1, 13, 25...)
        # En España las variables se suelen revisar anualmente
        if mes == 1 or (modo == "Variable" and (mes - 1) % 12 == 0) or (modo == "Fijo" and mes == 1):
            cuota_actual = calcular_cuota_mensual(saldo_pendiente, tasa_aplicable, plazo_meses - mes + 1)
            tasa_actual = tasa_aplicable

        interes_mes = saldo_pendiente * (tasa_actual / 12 / 100)
        amortizacion_capital = cuota_actual - interes_mes
        
        # Ajuste final
        if saldo_pendiente - amortizacion_capital < 0:
            amortizacion_capital = saldo_pendiente
            cuota_actual = interes_mes + amortizacion_capital

        saldo_pendiente -= amortizacion_capital
        
        calendario.append({
            "Mes": mes,
            "Año": ano_actual + 1,
            "Tipo Aplicado %": round(tasa_actual, 2),
            "Cuota": round(cuota_actual, 2),
            "Intereses": round(interes_mes, 2),
            "Amortización": round(amortizacion_capital, 2),
            "Capital Pendiente": round(max(0, saldo_pendiente), 2),
            "Fase": modo
        })
        
        if saldo_pendiente <= 0:
            break
            
    return pd.DataFrame(calendario)

# --- INTERFAZ DE USUARIO ---

st.title("🏡 Simulador de Hipotecas Avanzado")
st.markdown("Replica de análisis financiero tipo *Winvest* para hipotecas Mixtas y Variables.")

# Sidebar - Datos Generales
with st.sidebar:
    st.header("Datos del Préstamo")
    capital = st.number_input("Capital a solicitar (€)", min_value=10000, value=200000, step=1000)
    plazo = st.slider("Plazo (Años)", min_value=5, max_value=40, value=30)
    
    st.markdown("---")
    st.header("Escenario de Mercado")
    euribor_input = st.number_input("Euríbor Estimado Promedio (%)", min_value=-0.5, max_value=10.0, value=2.6, step=0.1)
    st.caption("Nota: Este simulador asume un Euríbor constante basado en tu proyección para simplificar la visualización a largo plazo.")

# Tabs para tipo de hipoteca
tab1, tab2 = st.tabs(["🔄 Hipoteca Variable", "🔀 Hipoteca Mixta"])

df_amortizacion = None
params = {}
tipo_seleccionado = ""

# --- LÓGICA HIPOTECA VARIABLE ---
with tab1:
    st.subheader("Configuración Variable")
    col1, col2 = st.columns(2)
    with col1:
        dif_var = st.number_input("Diferencial (%)", min_value=0.0, value=0.79, step=0.01, help="Lo que suma el banco al Euribor")
    with col2:
        st.info(f"Tipo Inicial Estimado: **{euribor_input + dif_var:.2f}%** (Euribor + Diferencial)")
    
    if st.button("Calcular Variable", type="primary"):
        tipo_seleccionado = "Variable"
        params = {"diferencial": dif_var, "euribor_proyeccion": euribor_input}
        df_amortizacion = generar_cuadro_amortizacion(capital, plazo, "Variable", params)

# --- LÓGICA HIPOTECA MIXTA ---
with tab2:
    st.subheader("Configuración Mixta")
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        periodo_fijo = st.number_input("Años a Tipo Fijo", min_value=1, max_value=plazo-1, value=5)
    with col_m2:
        tipo_fijo_mix = st.number_input("Tipo Fijo Inicial (%)", min_value=0.0, value=2.25, step=0.05)
    with col_m3:
        dif_mix = st.number_input("Diferencial tramo variable (%)", min_value=0.0, value=0.85, step=0.01)
        
    st.info(f"Resumen: **{periodo_fijo} años** al **{tipo_fijo_mix}%**, luego Euribor + **{dif_mix}%**")
    
    if st.button("Calcular Mixta", type="primary"):
        tipo_seleccionado = "Mixta"
        params = {
            "periodo_fijo": periodo_fijo, 
            "tipo_fijo": tipo_fijo_mix, 
            "diferencial": dif_mix,
            "euribor_proyeccion": euribor_input
        }
        df_amortizacion = generar_cuadro_amortizacion(capital, plazo, "Mixta", params)

# --- RESULTADOS Y VISUALIZACIÓN ---
if df_amortizacion is not None:
    st.markdown("---")
    st.header(f"📊 Análisis de Hipoteca {tipo_seleccionado}")
    
    # KPIs
    total_pagado = df_amortizacion['Cuota'].sum()
    total_intereses = df_amortizacion['Intereses'].sum()
    cuota_inicial = df_amortizacion.iloc[0]['Cuota']
    
    # Buscar si hay cambio de cuota (max cuota)
    cuota_max = df_amortizacion['Cuota'].max()
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Cuota Inicial", f"{cuota_inicial:,.2f} €")
    kpi2.metric("Cuota Máxima Estimada", f"{cuota_max:,.2f} €", delta=round(cuota_max - cuota_inicial, 2), delta_color="inverse")
    kpi3.metric("Total Intereses", f"{total_intereses:,.2f} €")
    kpi4.metric("Total a Pagar", f"{total_pagado:,.2f} €")
    
    # Gráficos
    st.subheader("Evolución Temporal")
    
    # Gráfico de Líneas (Capital vs Cuota)
    fig = go.Figure()
    
    # Eje Y primario: Cuota
    fig.add_trace(go.Scatter(
        x=df_amortizacion['Mes'], 
        y=df_amortizacion['Cuota'], 
        name='Cuota Mensual (€)',
        line=dict(color='firebrick', width=2)
    ))
    
    # Eje Y secundario: Capital Pendiente
    fig.add_trace(go.Scatter(
        x=df_amortizacion['Mes'], 
        y=df_amortizacion['Capital Pendiente'], 
        name='Capital Pendiente (€)',
        line=dict(color='royalblue', width=2, dash='dot'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Cuota Mensual vs Amortización del Capital',
        xaxis_title='Mes',
        yaxis=dict(title='Cuota Mensual (€)'),
        yaxis2=dict(title='Capital Pendiente (€)', overlaying='y', side='right'),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla Detallada
    with st.expander("Ver Cuadro de Amortización Completo"):
        st.dataframe(df_amortizacion.style.format({
            "Tipo Aplicado %": "{:.2f}%",
            "Cuota": "{:,.2f} €",
            "Intereses": "{:,.2f} €",
            "Amortización": "{:,.2f} €",
            "Capital Pendiente": "{:,.2f} €"
        }))
        
        # Botón de descarga
        csv = df_amortizacion.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar Cuadro en Excel (CSV)",
            data=csv,
            file_name=f'simulacion_hipoteca_{tipo_seleccionado.lower()}.csv',
            mime='text/csv',
        )
