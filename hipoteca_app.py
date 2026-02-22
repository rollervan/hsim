import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io

# ==========================================
# CONSTANTES (eliminados magic numbers)
# ==========================================
UMBRAL_SALDO = 1.0          # Saldo mínimo para considerar deuda pendiente
MAX_AMORT_ANUAL = 30_000    # Máxima amortización anticipada anual permitida
MESES_ANIO = 12

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="Simulador de Hipoteca PRO",
    page_icon="🏠",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
    h1, h2, h3 {font-family: sans-serif; color: #333;}
    .stMetric {background-color: #f9f9f9; border: 1px solid #ddd; padding: 15px; border-radius: 5px;}
    div[data-testid="stExpander"] {border: 1px solid #ddd; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)


# ==========================================
# UTILIDADES (extraídas al nivel de módulo)
# ==========================================
def fmt_t(m: int) -> str:
    """Formatea meses en texto legible (ej: 2a 3m)."""
    a = m // MESES_ANIO
    r = m % MESES_ANIO
    if a > 0 and r > 0:
        return f"{a}a {r}m"
    elif a > 0:
        return f"{a} años"
    else:
        return f"{r} meses"


def to_excel(df: pd.DataFrame) -> bytes:
    """Convierte un DataFrame a bytes de Excel."""
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Amortización')
            workbook = writer.book
            worksheet = writer.sheets['Amortización']
            fmt = workbook.add_format({'num_format': '#,##0.00'})
            worksheet.set_column('D:I', 12, fmt)
    except ModuleNotFoundError:
        with pd.ExcelWriter(output) as writer:
            df.to_excel(writer, index=False, sheet_name='Amortización')
    return output.getvalue()


def validar_inputs(capital, precio_vivienda, diferencial_a, diferencial_b, comparar) -> list[str]:
    """Valida los inputs y devuelve lista de advertencias."""
    warnings = []
    if precio_vivienda > 0 and capital > precio_vivienda:
        warnings.append("⚠️ La hipoteca supera el valor de la vivienda.")
    if capital <= 0:
        warnings.append("⚠️ El importe de la hipoteca debe ser mayor que 0.")
    if diferencial_a < 0:
        warnings.append("⚠️ El diferencial de la Opción A es negativo.")
    if comparar and diferencial_b < 0:
        warnings.append("⚠️ El diferencial de la Opción B es negativo.")
    return warnings


# ==========================================
# MOTOR DE CÁLCULO
# ==========================================
@st.cache_data(show_spinner=False)
def calcular_hipoteca_core(capital, anios, diferencial, tipo_fijo, anios_fijos, modo,
                           euribor_puntos, amortizaciones, tipo_reduc,
                           es_autopromotor, meses_carencia, apertura_pct, coste_cert):
    n_meses_total = int(anios * MESES_ANIO)

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

    puntos_eur = list(euribor_puntos) + [euribor_puntos[-1]] * (max(0, int(anios) - len(euribor_puntos)))
    len_amort = len(amortizaciones)
    puntos_amort = list(amortizaciones) + [0] * max(0, int(anios) - len_amort) if int(anios) > len_amort \
        else list(amortizaciones[:int(anios)])

    for anio in range(int(anios)):
        if modo == 'FIJA':
            tasa_anual = tipo_fijo
        elif modo == 'VARIABLE':
            tasa_anual = puntos_eur[anio] + diferencial
        else:  # MIXTA
            if anio < anios_fijos:
                tasa_anual = tipo_fijo
            else:
                tasa_anual = puntos_eur[anio] + diferencial

        tasa_mensual = (max(0, tasa_anual) / 100) / MESES_ANIO

        for m in range(MESES_ANIO):
            meses_restantes = n_meses_total - (mes_global - 1)
            en_periodo_carencia = es_autopromotor and (mes_global <= meses_carencia)

            gastos_fijos_mes = 0.0
            if mes_global == 1:
                gastos_fijos_mes += capital * (apertura_pct / 100)
            if en_periodo_carencia:
                gastos_fijos_mes += coste_cert

            if en_periodo_carencia:
                saldo_real += disposicion_mensual
                if saldo_real > capital:
                    saldo_real = capital
                cuota = saldo_real * tasa_mensual
                interes_m = cuota
                capital_m = 0
            else:
                if es_autopromotor and mes_global == meses_carencia + 1:
                    saldo_real = round(float(capital), 2)

                if saldo_real <= UMBRAL_SALDO:
                    saldo_real = 0
                    cuota = interes_m = capital_m = 0.0
                else:
                    base_calc = saldo_teorico if 'PLAZO' in tipo_reduc else saldo_real
                    if base_calc < saldo_real:
                        base_calc = saldo_real

                    if tasa_mensual > 0:
                        try:
                            cuota = base_calc * (tasa_mensual * (1 + tasa_mensual) ** meses_restantes) / \
                                    ((1 + tasa_mensual) ** meses_restantes - 1)
                        except Exception:
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

                    int_teorico = round(saldo_teorico * tasa_mensual, 2)
                    amort_teorica = round(cuota - int_teorico, 2)
                    saldo_teorico = round(max(0.0, saldo_teorico - amort_teorica), 2)

            data.append({
                'Mes': mes_global, 'Año': anio + 1, 'Tasa': tasa_anual if saldo_real > 0 else 0,
                'Cuota': cuota, 'Intereses': interes_m, 'Capital': capital_m,
                'Saldo': saldo_real, 'Amort_Extra': 0,
                'Gastos_Fijos': gastos_fijos_mes,
                'Fase': 'Carencia' if en_periodo_carencia else 'Amortización'
            })

            if not en_periodo_carencia:
                if m == MESES_ANIO - 1 and saldo_real > UMBRAL_SALDO and puntos_amort[anio] > 0:
                    ejec = round(min(puntos_amort[anio], saldo_real), 2)
                    saldo_real = round(saldo_real - ejec, 2)
                    if 'CUOTA' in tipo_reduc:
                        saldo_teorico = saldo_real
                    data[-1]['Amort_Extra'] = ejec
                    data[-1]['Capital'] = round(data[-1]['Capital'] + ejec, 2)
                    data[-1]['Saldo'] = saldo_real

            mes_global += 1

    return pd.DataFrame(data)


@st.cache_data(show_spinner=False)
def simular_vasicek(r0, theta, kappa, sigma, anios, n_sims=100, seed=None):
    """Simulación de Monte Carlo del Euríbor con modelo de Vasicek.
    Si seed no es None, fija la semilla para resultados reproducibles.
    """
    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(n_sims):
        camino = [r0]
        for _ in range(anios - 1):
            dr = kappa * (theta - camino[-1]) + sigma * rng.standard_normal()
            camino.append(max(-1.0, camino[-1] + dr))
        sims.append(camino)
    return np.array(sims)


@st.cache_data(show_spinner=False)
def calcular_cashflow(df_hipoteca, ingresos_mensuales, gastos_men_base,
                      gastos_anuales_base, ahorro_inicial, ipc, subida_salario):
    """Calcula el flujo de caja mensual y el ahorro acumulado."""
    df = df_hipoteca.copy()
    gasto_anual_mensualizado = gastos_anuales_base / MESES_ANIO

    ingresos_reales, gastos_vida_reales, coste_hipoteca_total = [], [], []
    flujos_mensuales, ahorro_acumulado = [], []
    saldo_actual = float(ahorro_inicial)

    for _, row in df.iterrows():
        mes_global = row['Mes']
        anio_actual = int((mes_global - 1) / MESES_ANIO)

        factor_ipc = (1 + ipc) ** anio_actual
        factor_salario = (1 + subida_salario) ** anio_actual

        ingreso_mes = ingresos_mensuales * factor_salario
        gasto_vida_mes = (gastos_men_base + gasto_anual_mensualizado) * factor_ipc

        cuota = float(row['Cuota'])
        seguros = float(row.get('Seguros', 0))
        gastos_fijos = float(row['Gastos_Fijos'])
        amort_extra = float(row['Amort_Extra'])
        coste_hip_mes = cuota + seguros + gastos_fijos + amort_extra

        flujo_neto = ingreso_mes - gasto_vida_mes - coste_hip_mes
        saldo_actual += flujo_neto

        ingresos_reales.append(ingreso_mes)
        gastos_vida_reales.append(gasto_vida_mes)
        coste_hipoteca_total.append(coste_hip_mes)
        flujos_mensuales.append(flujo_neto)
        ahorro_acumulado.append(saldo_actual)

    df['Ingresos_Real'] = ingresos_reales
    df['Gastos_Vida_Real'] = gastos_vida_reales
    df['Coste_Hip_Total'] = coste_hipoteca_total
    df['Flujo_Mensual'] = flujos_mensuales
    df['Ahorro_Disponible'] = ahorro_acumulado
    return df


def ajustar_amortizaciones(capital: float, anios: int, diferencial: float, tipo_fijo: float,
                           anios_fijos: int, modo: str, euribor_puntos,
                           amortizaciones: list, tipo_reduc: str,
                           es_autopromotor: bool, meses_carencia: int) -> list:
    """
    Recorre el plan de amortización año a año estimando el saldo pendiente al final
    de cada año (solo cuotas ordinarias, sin amortizaciones extra aún).
    Cuando la amortización programada de un año supera el saldo real disponible,
    la recorta al máximo posible y pone a cero todos los años siguientes.
    Devuelve la lista de amortizaciones corregida.
    """
    # Estimación rápida del saldo al final de cada año usando solo cuotas ordinarias
    saldo = float(capital) if not es_autopromotor else float(capital)  # peor caso: capital completo
    puntos_eur = list(euribor_puntos) + [euribor_puntos[-1]] * max(0, anios - len(euribor_puntos))
    amort_ajustada = []
    hipoteca_saldada = False

    for anio in range(anios):
        if hipoteca_saldada or saldo <= UMBRAL_SALDO:
            amort_ajustada.append(0)
            continue

        # Tasa del año
        if modo == 'FIJA':
            tasa_anual = tipo_fijo
        elif modo == 'VARIABLE':
            tasa_anual = puntos_eur[anio] + diferencial
        else:
            tasa_anual = tipo_fijo if anio < anios_fijos else puntos_eur[anio] + diferencial
        tasa_mensual = (max(0, tasa_anual) / 100) / MESES_ANIO

        # Amortización ordinaria del año (12 cuotas)
        meses_restantes_inicio = (anios - anio) * MESES_ANIO
        for _ in range(MESES_ANIO):
            if saldo <= UMBRAL_SALDO:
                break
            if tasa_mensual > 0:
                try:
                    cuota = saldo * (tasa_mensual * (1 + tasa_mensual) ** meses_restantes_inicio) / \
                            ((1 + tasa_mensual) ** meses_restantes_inicio - 1)
                except Exception:
                    cuota = saldo / meses_restantes_inicio
            else:
                cuota = saldo / meses_restantes_inicio
            interes_m = saldo * tasa_mensual
            capital_m = min(cuota - interes_m, saldo)
            saldo = round(max(0.0, saldo - capital_m), 2)
            meses_restantes_inicio -= 1

        # Amortización extra del año
        solicitada = amortizaciones[anio] if anio < len(amortizaciones) else 0
        if solicitada <= 0 or saldo <= UMBRAL_SALDO:
            amort_ajustada.append(0)
        elif solicitada >= saldo:
            # Recortar al saldo disponible y marcar como saldada
            amort_ajustada.append(round(saldo, 2))
            saldo = 0.0
            hipoteca_saldada = True
        else:
            amort_ajustada.append(solicitada)
            saldo = round(saldo - solicitada, 2)

    return amort_ajustada


def agregar_por_anio(df: pd.DataFrame) -> pd.DataFrame:
    """Genera un resumen anual del cuadro de amortización."""
    return df.groupby('Año').agg(
        Cuota_Media=('Cuota', 'mean'),
        Intereses=('Intereses', 'sum'),
        Capital_Amortizado=('Capital', 'sum'),
        Amort_Extra=('Amort_Extra', 'sum'),
        Seguros=('Seguros', 'sum'),
        Gastos_Fijos=('Gastos_Fijos', 'sum'),
        Saldo_Final=('Saldo', 'last'),
        Tasa=('Tasa', 'mean'),
    ).reset_index()


# ==========================================
# WIDGET REUTILIZABLE: PESTAÑA LIQUIDEZ
# ==========================================
def mostrar_liquidez(df_a: pd.DataFrame, df_b: pd.DataFrame | None, comparar: bool):
    st.subheader("💰 Análisis de Liquidez")

    st.markdown("##### 1. Ahorro Mensual (Lo que te sobra cada mes)")
    st.caption("Verde: Ingresas más de lo que gastas. Rojo: Gastas más de lo que ingresas.")

    fig_mes = go.Figure()
    colors_a = ['#2ca02c' if v >= 0 else '#d62728' for v in df_a['Flujo_Mensual']]
    fig_mes.add_trace(go.Bar(x=df_a['Mes'], y=df_a['Flujo_Mensual'],
                             marker_color=colors_a, name='Ahorro Mes A'))

    if comparar and df_b is not None:
        fig_mes.add_trace(go.Scatter(x=df_b['Mes'], y=df_b['Flujo_Mensual'],
                                     mode='lines', line=dict(color='#ff7f0e', width=2, dash='dot'),
                                     name='Ahorro Mes B'))

    fig_mes.update_layout(height=350, template="plotly_white",
                          yaxis_title="Euros (€)", xaxis_title="Mes", hovermode="x unified")
    st.plotly_chart(fig_mes, use_container_width=True)

    st.divider()
    st.markdown("##### 2. Saldo Acumulado en Cuenta (Colchón)")
    st.caption("Cómo evoluciona tu dinero total en el banco a lo largo de los años.")

    saldo_final_a = df_a['Ahorro_Disponible'].iloc[-1]
    min_liquidez_a = df_a['Ahorro_Disponible'].min()
    c1, c2 = st.columns(2)
    c1.metric("Saldo Final (Año Fin)", f"{saldo_final_a:,.0f} €")
    c2.metric("Momento más crítico", f"{min_liquidez_a:,.0f} €",
              delta="Peligro" if min_liquidez_a < 0 else "Ok", delta_color="inverse")

    fig_cf = go.Figure()
    fig_cf.add_trace(go.Scatter(x=df_a['Mes'], y=df_a['Ahorro_Disponible'],
                                name='Saldo A', fill='tozeroy', line=dict(color='#1f77b4')))
    if comparar and df_b is not None:
        fig_cf.add_trace(go.Scatter(x=df_b['Mes'], y=df_b['Ahorro_Disponible'],
                                    name='Saldo B', line=dict(color='#ff7f0e', dash='dash')))
    fig_cf.add_hline(y=0, line_dash="dot", line_color="red", annotation_text="0 €")
    fig_cf.update_layout(height=350, template="plotly_white", yaxis_title="Saldo Total (€)")
    st.plotly_chart(fig_cf, use_container_width=True)

    st.divider()
    st.subheader("🕵️‍♀️ Verificación de Cálculos")
    st.caption("Revisa los números exactos mes a mes.")

    cols_check = ['Mes', 'Año', 'Ingresos_Real', 'Gastos_Vida_Real', 'Cuota',
                  'Seguros', 'Gastos_Fijos', 'Flujo_Mensual', 'Ahorro_Disponible']

    # Filtro por año para no mostrar 300 filas de golpe
    anio_sel = st.selectbox("Filtrar por año:", ['Todos'] + list(df_a['Año'].unique()),
                            key=f"liq_anio_{comparar}")
    df_show = df_a if anio_sel == 'Todos' else df_a[df_a['Año'] == anio_sel]
    st.dataframe(df_show[cols_check].style.format("{:,.2f} €"),
                 use_container_width=True, height=350)

    st.info("""
    **Fórmula:** Flujo = Ingresos − (Gastos Vida + Gastos Anuales/12) − (Cuota + Seguros + Gastos Fijos)

    *Si el Mes 1 muestra un ahorro muy negativo, suele deberse a la Comisión de Apertura en 'Gastos_Fijos'.*
    """)


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("Configuración")
    comparar = st.checkbox("Comparar dos opciones", value=False)

    with st.expander("Datos Económicos y Proyecto", expanded=True):
        ingresos = st.number_input("Ingresos netos (€)", value=2500, step=100)
        ahorro_inicial = st.number_input("Ahorro inicial (€)", value=0, step=1000)
        precio_vivienda = st.number_input("Valor Vivienda (€)", value=0, step=5000)
        capital_init_global = st.number_input("Importe Hipoteca (€)", value=180000, step=1000)

        st.markdown("---")
        es_autopromotor = st.checkbox("Es Autopromoción (Obra)", value=True)
        meses_carencia = 0
        if es_autopromotor:
            meses_carencia = st.number_input("Meses de Carencia", value=11, min_value=1, max_value=36)
            st.caption("Durante la carencia pagas intereses y certificaciones.")

    st.markdown("---")

    if comparar:
        st.subheader("Opción A vs Opción B")
        colA, colB = st.columns(2)

        with colA:
            st.markdown("#### Opción A")
            modo_A = st.selectbox("Tipo A", ["MIXTA", "VARIABLE", "FIJA"], key="mA")
            anios_A = st.number_input("Años A", value=25, key="yA")
            tipo_fijo_A = diferencial_A = 0.0
            anios_fijos_A = 0
            if modo_A == "FIJA":
                tipo_fijo_A = st.number_input("TIN A (%)", value=2.15, step=0.05, key="tfA")
            elif modo_A == "VARIABLE":
                diferencial_A = st.number_input("Dif. A (%)", value=0.55, step=0.05, key="dfA")
            elif modo_A == "MIXTA":
                tipo_fijo_A = st.number_input("Fijo A (%)", value=2.2, step=0.05, key="mfaA")
                anios_fijos_A = st.number_input("Años Fijos A", value=7, key="myaA")
                diferencial_A = st.number_input("Dif. Var A", value=0.55, step=0.05, key="mdaA")
            st.caption("Seguros y Gastos A")
            s_hogar_A = st.number_input("Hogar A (€/año)", value=280, key="shA")
            s_vida_A = st.number_input("Vida A (€/año)", value=188, key="svA")
            apertura_A = st.number_input("Apertura A (%)", value=0.4, step=0.1, key="apA")
            cert_A = st.number_input("Certif. A (€/mes)", value=30.0, key="ctA") if es_autopromotor else 0.0

        with colB:
            st.markdown("#### Opción B")
            modo_B = st.selectbox("Tipo B", ["MIXTA", "VARIABLE", "FIJA"], index=2, key="mB")
            anios_B = st.number_input("Años B", value=25, key="yB")
            tipo_fijo_B = diferencial_B = 0.0
            anios_fijos_B = 0
            if modo_B == "FIJA":
                tipo_fijo_B = st.number_input("TIN B (%)", value=2.15, step=0.05, key="tfB")
            elif modo_B == "VARIABLE":
                diferencial_B = st.number_input("Dif. B (%)", value=0.55, step=0.05, key="dfB")
            elif modo_B == "MIXTA":
                tipo_fijo_B = st.number_input("Fijo B (%)", value=2.2, step=0.05, key="mfaB")
                anios_fijos_B = st.number_input("Años Fijos B", value=7, key="myaB")
                diferencial_B = st.number_input("Dif. Var B", value=0.55, step=0.05, key="mdaB")
            st.caption("Seguros y Gastos B")
            s_hogar_B = st.number_input("Hogar B (€/año)", value=380, key="shB")
            s_vida_B = st.number_input("Vida B (€/año)", value=384, key="svB")
            apertura_B = st.number_input("Apertura B (%)", value=0.0, step=0.1, key="apB")
            cert_B = st.number_input("Certif. B (€/mes)", value=200.0, key="ctB") if es_autopromotor else 0.0

    else:
        st.subheader("Condiciones Préstamo")
        modo_A = st.selectbox("Modalidad", ["MIXTA", "VARIABLE", "FIJA"])
        anios_A = st.number_input("Plazo (Años)", value=25, min_value=1)
        tipo_fijo_A = diferencial_A = 0.0
        anios_fijos_A = 0
        c1, c2 = st.columns(2)
        if modo_A == "FIJA":
            tipo_fijo_A = c1.number_input("TIN Fijo (%)", value=2.15, step=0.05)
        elif modo_A == "VARIABLE":
            diferencial_A = c1.number_input("Diferencial (%)", value=0.55, step=0.05)
        elif modo_A == "MIXTA":
            tipo_fijo_A = c1.number_input("Fijo (%)", value=2.2, step=0.05)
            anios_fijos_A = c2.number_input("Años Fijos", value=7)
            diferencial_A = st.number_input("Dif. Variable (%)", value=0.55, step=0.05)

        st.markdown("**Seguros y Gastos**")
        s_hogar_A = st.number_input("Seguro Hogar (€/año)", value=280)
        s_vida_A = st.number_input("Seguro Vida (€/año)", value=188)
        apertura_A = st.number_input("Comisión Apertura (%)", value=0.0, step=0.1)
        cert_A = st.number_input("Coste Certificación (€/mes)", value=30.0) if es_autopromotor else 0.0

        modo_B, anios_B = modo_A, anios_A
        tipo_fijo_B, diferencial_B, anios_fijos_B = tipo_fijo_A, diferencial_A, anios_fijos_A
        s_hogar_B, s_vida_B = s_hogar_A, s_vida_A
        apertura_B, cert_B = apertura_A, cert_A

    st.markdown("---")
    tipo_reduc = st.radio("Amortización anticipada:", ["Reducir PLAZO", "Reducir CUOTA"])

    with st.expander("Gastos de Vida y Ajustes", expanded=False):
        g_comida = st.number_input("Comida", value=200)
        g_suministros = st.number_input("Suministros (Luz/Agua/Internet)", value=100)
        g_gasolina = st.number_input("Transporte", value=120)
        g_otros = st.number_input("Ocio y Otros", value=150)
        g_anuales = st.number_input("Gastos Anuales (Vacaciones, IBI...)", value=3500, step=500)
        ipc = st.slider("IPC Estimado (%)", 0.0, 10.0, 0.0, step=0.1) / 100
        subida_salarial = st.slider("Subida Salarial Anual %", 0.0, 10.0, 0.0, step=0.1) / 100

    # Previsión Euríbor
    max_anios = max(anios_A, anios_B)
    caminos_eur = []
    n_sims = 1
    necesita_euribor = (modo_A != "FIJA") or (comparar and modo_B != "FIJA")

    if necesita_euribor:
        st.markdown("---")
        with st.expander("Previsión Euríbor", expanded=True):
            modo_prev = st.selectbox("Método", ["Monte Carlo (Simulación)", "Manual"])
            if modo_prev == "Monte Carlo (Simulación)":
                n_sims = st.select_slider("Simulaciones", [50, 100, 500, 1000], value=100)
                seed_fija = st.checkbox("🔒 Seed fija (resultados reproducibles)", value=True)
                st.caption("Ajustes Mercado")
                theta = st.slider("Media L/P", 0.0, 5.0, 2.25)
                sigma = st.slider("Volatilidad", 0.0, 2.0, 0.60)
                kappa = st.slider("Reversión", 0.0, 1.0, 0.30)
                r0 = st.number_input("Euríbor Hoy", value=2.24)
            else:
                n_sims = 1
                seed_fija = True

            if modo_prev == "Manual":
                eur_list = []
                cols_eur = st.columns(5)
                for i in range(max_anios):
                    with cols_eur[i % 5]:
                        eur_list.append(st.number_input(f"A{i+1}", value=2.2, step=0.1, key=f"e{i}"))
                caminos_eur = [eur_list]
            else:
                caminos_eur = simular_vasicek(
                    r0, theta, kappa, sigma, max_anios, n_sims,
                    seed=42 if seed_fija else None
                )
    else:
        caminos_eur = [[0.0] * max_anios]
        n_sims = 1
        seed_fija = True


# ==========================================
# VALIDACIONES DE INPUTS
# ==========================================
st.title("Simulador de Hipoteca PRO")

alertas = validar_inputs(
    capital_init_global, precio_vivienda,
    diferencial_A, diferencial_B if comparar else 0.0, comparar
)
for alerta in alertas:
    st.warning(alerta)


# ==========================================
# AMORTIZACIÓN ANTICIPADA
# ==========================================
# Calculamos el saldo de referencia para ajustar los sliders ANTES de renderizarlos.
# Usamos el camino mediano del Euríbor como referencia.
camino_ref = tuple(caminos_eur[len(caminos_eur) // 2]) if len(caminos_eur) > 1 else tuple(caminos_eur[0])

# Leemos los valores actuales de session_state (o 0 si no existen aún)
amort_bruta = [st.session_state.get(f"s_a{i}", 0) for i in range(max_anios)]

# Calculamos la versión ajustada para la opción A (usada como referencia para los sliders)
amort_list_A_preview = ajustar_amortizaciones(
    capital_init_global, anios_A, diferencial_A, tipo_fijo_A, anios_fijos_A,
    modo_A, camino_ref, amort_bruta, tipo_reduc, es_autopromotor, meses_carencia
)

# Si algún valor fue recortado, forzamos el session_state al valor ajustado
# ANTES de que los sliders se rendericen en esta pasada.
for i, (orig, adj) in enumerate(zip(amort_bruta, amort_list_A_preview)):
    if orig != adj:
        st.session_state[f"s_a{i}"] = adj

with st.expander("Amortización Anticipada"):
    st.info(f"Capital extra anual (Máx. {MAX_AMORT_ANUAL:,} €)")
    hubo_ajuste = any(o != a for o, a in zip(amort_bruta, amort_list_A_preview))
    if hubo_ajuste:
        años_ajustados = [i+1 for i, (o, a) in enumerate(zip(amort_bruta, amort_list_A_preview)) if o != a]
        años_texto = ", ".join(f"Año {a}" for a in años_ajustados)
        st.info(
            f"ℹ️ **Sliders ajustados automáticamente** en **{años_texto}**: "
            f"el importe solicitado superaba el saldo pendiente, así que se ha recortado "
            f"al máximo amortizable y los años posteriores se han puesto a 0."
        )
    cols_a = st.columns(4)
    amort_list = []
    for i in range(max_anios):
        val = cols_a[i % 4].slider(f"Año {i+1}", 0, MAX_AMORT_ANUAL, 0, step=500, key=f"s_a{i}")
        amort_list.append(val)

hay_amortizacion = sum(amort_list) > 0

# Ajuste final con los valores definitivos de los sliders (ya corregidos)
amort_list_A = ajustar_amortizaciones(
    capital_init_global, anios_A, diferencial_A, tipo_fijo_A, anios_fijos_A,
    modo_A, camino_ref, amort_list, tipo_reduc, es_autopromotor, meses_carencia
)
amort_list_B = ajustar_amortizaciones(
    capital_init_global, anios_B, diferencial_B, tipo_fijo_B, anios_fijos_B,
    modo_B, camino_ref, amort_list, tipo_reduc, es_autopromotor, meses_carencia
) if comparar else amort_list_A

# ==========================================
# BUCLE DE SIMULACIÓN
# ==========================================
kpis_int_A, kpis_int_B = [], []
kpis_pat_A, kpis_ahorro_A = [], []
eur_matrix = []

df_median_A = df_median_B = None
df_base_median_A = None

total_gastos = g_comida + g_suministros + g_gasolina + g_otros
coste_mes_seguros_A = (s_hogar_A + s_vida_A) / MESES_ANIO
coste_mes_seguros_B = (s_hogar_B + s_vida_B) / MESES_ANIO

if n_sims > 100:
    prog_bar = st.progress(0)

for i, camino in enumerate(caminos_eur):
    camino_tuple = tuple(camino)  # hashable para cache

    df_A = calcular_hipoteca_core(
        capital_init_global, anios_A, diferencial_A, tipo_fijo_A, anios_fijos_A,
        modo_A, camino_tuple, tuple(amort_list_A), tipo_reduc,
        es_autopromotor, meses_carencia, apertura_A, cert_A
    )

    # Escenario base (sin amortización) — solo calculado si modo individual
    if not comparar:
        df_base_A = calcular_hipoteca_core(
            capital_init_global, anios_A, diferencial_A, tipo_fijo_A, anios_fijos_A,
            modo_A, camino_tuple, tuple([0] * anios_A), 'PLAZO',
            es_autopromotor, meses_carencia, apertura_A, cert_A
        )
        kpis_ahorro_A.append(df_base_A['Intereses'].sum() - df_A['Intereses'].sum())

    df_A['Seguros'] = np.where(df_A['Saldo'] > 0, coste_mes_seguros_A, 0)
    gasto_tot_A = df_A['Cuota'] + df_A['Seguros'] + total_gastos
    df_A['Ahorro_Liq'] = ahorro_inicial + (ingresos - gasto_tot_A).cumsum() \
                         - df_A['Amort_Extra'].cumsum() - df_A['Gastos_Fijos'].cumsum()
    df_A['Patrimonio'] = df_A['Ahorro_Liq'] + (precio_vivienda - df_A['Saldo'])

    kpis_int_A.append(df_A['Intereses'].sum() + df_A['Seguros'].sum() + df_A['Gastos_Fijos'].sum())

    if not comparar:
        kpis_pat_A.append(df_A['Patrimonio'].iloc[-1])
        eur_matrix.append(camino)

    if comparar:
        df_B = calcular_hipoteca_core(
            capital_init_global, anios_B, diferencial_B, tipo_fijo_B, anios_fijos_B,
            modo_B, camino_tuple, tuple(amort_list_B), tipo_reduc,
            es_autopromotor, meses_carencia, apertura_B, cert_B
        )
        df_B['Seguros'] = np.where(df_B['Saldo'] > 0, coste_mes_seguros_B, 0)
        kpis_int_B.append(df_B['Intereses'].sum() + df_B['Seguros'].sum() + df_B['Gastos_Fijos'].sum())
        if i == 0:
            df_median_B = df_B

    if i == 0:
        df_median_A = df_A
        if not comparar:
            df_base_median_A = df_base_A

    if n_sims > 100:
        prog_bar.progress((i + 1) / n_sims)

if n_sims > 100:
    prog_bar.empty()


# ==========================================
# ESCENARIO MEDIANO
# ==========================================
idx_med = np.argsort(kpis_int_A)[len(kpis_int_A) // 2]
if n_sims > 1:
    camino_med = tuple(caminos_eur[idx_med])

    df_median_A = calcular_hipoteca_core(
        capital_init_global, anios_A, diferencial_A, tipo_fijo_A, anios_fijos_A,
        modo_A, camino_med, tuple(amort_list_A), tipo_reduc,
        es_autopromotor, meses_carencia, apertura_A, cert_A
    )
    df_median_A['Seguros'] = np.where(df_median_A['Saldo'] > 0, coste_mes_seguros_A, 0)
    df_median_A['Ahorro_Liq'] = ahorro_inicial + (
        ingresos - (df_median_A['Cuota'] + df_median_A['Seguros'] + total_gastos)
    ).cumsum() - df_median_A['Amort_Extra'].cumsum() - df_median_A['Gastos_Fijos'].cumsum()
    df_median_A['Patrimonio'] = df_median_A['Ahorro_Liq'] + (precio_vivienda - df_median_A['Saldo'])

    if not comparar:
        df_base_median_A = calcular_hipoteca_core(
            capital_init_global, anios_A, diferencial_A, tipo_fijo_A, anios_fijos_A,
            modo_A, camino_med, tuple([0] * anios_A), 'PLAZO',
            es_autopromotor, meses_carencia, apertura_A, cert_A
        )

    if comparar:
        df_median_B = calcular_hipoteca_core(
            capital_init_global, anios_B, diferencial_B, tipo_fijo_B, anios_fijos_B,
            modo_B, camino_med, tuple(amort_list_B), tipo_reduc,
            es_autopromotor, meses_carencia, apertura_B, cert_B
        )
        df_median_B['Seguros'] = np.where(df_median_B['Saldo'] > 0, coste_mes_seguros_B, 0)


# ==========================================
# CASHFLOW (una sola vez por escenario mediano)
# ==========================================
total_gastos_mensuales = g_comida + g_suministros + g_gasolina + g_otros

df_median_A = calcular_cashflow(
    df_median_A, ingresos, total_gastos_mensuales, g_anuales, ahorro_inicial, ipc, subida_salarial
)

if comparar and df_median_B is not None:
    df_median_B = calcular_cashflow(
        df_median_B, ingresos, total_gastos_mensuales, g_anuales, ahorro_inicial, ipc, subida_salarial
    )


# ==========================================
# KPIs GENERALES
# ==========================================
coste_A = df_median_A['Intereses'].sum() + df_median_A['Seguros'].sum() + df_median_A['Gastos_Fijos'].sum()
meses_reales_A = len(df_median_A[df_median_A['Saldo'] > UMBRAL_SALDO])

idx_ref = meses_carencia if (es_autopromotor and not comparar) else 0
if idx_ref >= len(df_median_A):
    idx_ref = 0
cuota_ini_A = df_median_A.iloc[idx_ref]['Cuota']

cols_tabla = ['Año', 'Mes', 'Tasa', 'Cuota', 'Intereses', 'Capital',
              'Amort_Extra', 'Gastos_Fijos', 'Saldo']


# ==========================================
# RESULTADOS
# ==========================================

if comparar:
    coste_B = df_median_B['Intereses'].sum() + df_median_B['Seguros'].sum() + df_median_B['Gastos_Fijos'].sum()
    meses_reales_B = len(df_median_B[df_median_B['Saldo'] > UMBRAL_SALDO])
    cuota_ini_B = df_median_B.iloc[0]['Cuota']

    st.markdown("### 🆚 Resultados Comparativa")

    dif_coste = coste_B - coste_A
    ahorro = abs(dif_coste)

    if dif_coste > 1000:
        st.success(f"🏆 **La Opción A es mejor**: Te ahorras **{ahorro:,.0f} €** respecto a la B.")
    elif dif_coste < -1000:
        st.success(f"🏆 **La Opción B es mejor**: Te ahorras **{ahorro:,.0f} €** respecto a la A.")
    else:
        st.info("⚖️ **Empate técnico**: La diferencia es menor a 1.000 €.")

    st.markdown("---")
    k1, k2, k3 = st.columns(3)
    k1.metric("Opción A", f"{coste_A:,.0f} €")
    k2.metric("Opción B", f"{coste_B:,.0f} €")
    if dif_coste > 0:
        k3.metric("Diferencia", f"{ahorro:,.0f} €", "Ahorro con A", delta_color="normal")
    else:
        k3.metric("Diferencia", f"{ahorro:,.0f} €", "Ahorro con B", delta_color="normal")

    st.markdown("<br>", unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    t1.metric("Plazo Real A", fmt_t(meses_reales_A))
    t2.metric("Plazo Real B", fmt_t(meses_reales_B))
    dif_meses = meses_reales_B - meses_reales_A
    if dif_meses > 0:
        t3.metric("Diferencia Tiempo", f"{abs(dif_meses)} meses", "A termina antes")
    elif dif_meses < 0:
        t3.metric("Diferencia Tiempo", f"{abs(dif_meses)} meses", "B termina antes")
    else:
        t3.metric("Diferencia Tiempo", "Igual", "Mismo plazo", delta_color="off")

    st.markdown("---")

    tabs = st.tabs(["Evolución Deuda", "Costes Acumulados", "Tabla de Datos",
                    "Resumen Anual", "Análisis de Riesgo", "💰 Liquidez"])

    with tabs[0]:
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=df_median_A['Mes'], y=df_median_A['Saldo'],
                                   fill='tozeroy', name='Opción A', line=dict(color='#0055aa')))
        fig_s.add_trace(go.Scatter(x=df_median_B['Mes'], y=df_median_B['Saldo'],
                                   name='Opción B', line=dict(color='#ff7f0e', dash='dash', width=3)))
        st.plotly_chart(fig_s, use_container_width=True)

    with tabs[1]:
        c_a1, c_a2 = st.columns(2)
        with c_a1:
            st.subheader("Desglose de Costes Acumulados")
            fig_i = go.Figure()
            for df_, color, dash, label in [
                (df_median_A, '#0055aa', 'solid', 'A'),
                (df_median_B, '#ff7f0e', 'dash', 'B'),
            ]:
                fig_i.add_trace(go.Scatter(
                    x=df_['Mes'],
                    y=(df_['Intereses'] + df_['Seguros'] + df_['Gastos_Fijos']).cumsum(),
                    name=f'TOTAL {label}', line=dict(color=color, width=3, dash=dash)))
                fig_i.add_trace(go.Scatter(
                    x=df_['Mes'], y=df_['Intereses'].cumsum(),
                    name=f'Intereses {label}', line=dict(color=color, width=1.5, dash=dash)))
                fig_i.add_trace(go.Scatter(
                    x=df_['Mes'], y=df_['Seguros'].cumsum(),
                    name=f'Seguros {label}', line=dict(color=color, width=1.5, dash='dot')))
            fig_i.update_layout(template='plotly_white', height=450,
                                 legend=dict(orientation="h", y=1.1), hovermode="x unified",
                                 yaxis_title="Euros Acumulados (€)")
            st.plotly_chart(fig_i, use_container_width=True)

        with c_a2:
            st.subheader("Cuota Mensual")
            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(x=df_median_A['Mes'], y=df_median_A['Cuota'],
                                       name='Cuota A', line=dict(color='#0055aa')))
            fig_c.add_trace(go.Scatter(x=df_median_B['Mes'], y=df_median_B['Cuota'],
                                       name='Cuota B', line=dict(color='#ff7f0e', dash='dash')))
            if es_autopromotor:
                fig_c.add_vline(x=meses_carencia, line_dash="dot",
                                annotation_text="Fin Carencia", line_color="gray")
            fig_c.update_layout(template='plotly_white', height=450, legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_c, use_container_width=True)

    with tabs[2]:
        st.subheader("Tablas de Amortización Detalladas")
        col_tbl1, col_tbl2 = st.columns(2)
        with col_tbl1:
            st.markdown("#### 🔹 Opción A")
            df_A_show = df_median_A[cols_tabla].copy()
            st.download_button("📥 Excel A", data=to_excel(df_A_show), file_name='hipoteca_A.xlsx')
            st.dataframe(df_A_show.style.format(
                {'Tasa': '{:.2f}%', 'Cuota': '{:,.0f}', 'Saldo': '{:,.0f}', 'Gastos_Fijos': '{:,.0f}'}),
                height=300, use_container_width=True)
        with col_tbl2:
            st.markdown("#### 🔸 Opción B")
            df_B_show = df_median_B[cols_tabla].copy()
            st.download_button("📥 Excel B", data=to_excel(df_B_show), file_name='hipoteca_B.xlsx')
            st.dataframe(df_B_show.style.format(
                {'Tasa': '{:.2f}%', 'Cuota': '{:,.0f}', 'Saldo': '{:,.0f}', 'Gastos_Fijos': '{:,.0f}'}),
                height=300, use_container_width=True)

    with tabs[3]:
        st.subheader("Resumen Anual")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("#### 🔹 Opción A")
            anual_a = agregar_por_anio(df_median_A)
            st.download_button("📥 Excel Anual A", data=to_excel(anual_a), file_name='anual_A.xlsx')
            st.dataframe(anual_a.style.format(
                {'Cuota_Media': '{:,.0f}', 'Intereses': '{:,.0f}', 'Capital_Amortizado': '{:,.0f}',
                 'Saldo_Final': '{:,.0f}', 'Tasa': '{:.2f}%', 'Seguros': '{:,.0f}',
                 'Gastos_Fijos': '{:,.0f}', 'Amort_Extra': '{:,.0f}'}),
                height=350, use_container_width=True)
        with col_r2:
            st.markdown("#### 🔸 Opción B")
            anual_b = agregar_por_anio(df_median_B)
            st.download_button("📥 Excel Anual B", data=to_excel(anual_b), file_name='anual_B.xlsx')
            st.dataframe(anual_b.style.format(
                {'Cuota_Media': '{:,.0f}', 'Intereses': '{:,.0f}', 'Capital_Amortizado': '{:,.0f}',
                 'Saldo_Final': '{:,.0f}', 'Tasa': '{:.2f}%', 'Seguros': '{:,.0f}',
                 'Gastos_Fijos': '{:,.0f}', 'Amort_Extra': '{:,.0f}'}),
                height=350, use_container_width=True)

    with tabs[4]:
        st.subheader("Distribución de Coste Total")
        if n_sims < 10:
            st.warning("⚠️ Selecciona 'Monte Carlo' con +50 simulaciones para ver el riesgo.")
        else:
            fig_risk = go.Figure()
            fig_risk.add_trace(go.Histogram(x=kpis_int_A, name='Opción A', opacity=0.75, marker_color='#0055aa'))
            fig_risk.add_trace(go.Histogram(x=kpis_int_B, name='Opción B', opacity=0.75, marker_color='#ff7f0e'))
            fig_risk.update_layout(barmode='overlay', title_text='Probabilidad de Coste',
                                   xaxis_title_text='Coste Total', template='plotly_white')
            st.plotly_chart(fig_risk, use_container_width=True)

    with tabs[5]:
        mostrar_liquidez(df_median_A, df_median_B, comparar=True)

else:
    # ─── VISTA INDIVIDUAL ───
    if hay_amortizacion:
        meses_base = len(df_base_median_A[df_base_median_A['Saldo'] > UMBRAL_SALDO])
        meses_actual = len(df_median_A[df_median_A['Saldo'] > UMBRAL_SALDO])
        meses_ahorrados = max(0, meses_base - meses_actual)
        txt_duracion = fmt_t(meses_actual)
        a_save = meses_ahorrados // MESES_ANIO
        m_save = meses_ahorrados % MESES_ANIO
        txt_tiempo = f"-{a_save}a {m_save}m" if 'PLAZO' in tipo_reduc and meses_ahorrados > 0 else "Baja cuota"
        ahorro_int = np.median(kpis_ahorro_A)
    else:
        meses_base = len(df_median_A[df_median_A['Saldo'] > UMBRAL_SALDO])
        txt_duracion = fmt_t(meses_base)
        txt_tiempo = "Sin cambios"
        ahorro_int = 0

    st.markdown("### Resumen")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cuota Inicial", f"{cuota_ini_A:,.2f} €", f"{df_median_A.iloc[idx_ref]['Tasa']:.2f}% TIN")

    val_int = df_median_A['Intereses'].sum()
    val_seg = df_median_A['Seguros'].sum()
    val_gas = df_median_A['Gastos_Fijos'].sum()
    val_tot = val_int + val_seg + val_gas

    k2.metric("Coste Total", f"{val_tot:,.0f} €", delta_color="inverse")
    k2.caption(f"Int: {val_int:,.0f} | Seg: {val_seg:,.0f} | Gas: {val_gas:,.0f}")
    k3.metric("Plazo Final", txt_duracion, delta_color="off")
    k4.metric("Ahorro", f"{ahorro_int:,.0f} €", txt_tiempo)

    st.markdown("---")

    tabs = st.tabs(["Evolución", "Amortización", "Patrimonio", "Resumen Anual", "Riesgo", "💰 Liquidez"])

    with tabs[0]:
        c_e1, c_e2 = st.columns(2)
        with c_e1:
            st.subheader("Euríbor Estimado")
            if modo_A == "FIJA":
                fig_fija = go.Figure()
                fig_fija.add_shape(type="line", x0=0, x1=anios_A, y0=tipo_fijo_A, y1=tipo_fijo_A,
                                   line=dict(color='#0055aa', width=3, dash='dot'))
                fig_fija.add_annotation(x=anios_A / 2, y=tipo_fijo_A,
                                        text=f"Tipo Fijo: {tipo_fijo_A:.2f}%",
                                        showarrow=False, yshift=15,
                                        font=dict(size=14, color='#0055aa'))
                fig_fija.update_layout(template='plotly_white', height=350,
                                       yaxis_title="Tipo (%)", xaxis_title="Año",
                                       yaxis_range=[0, tipo_fijo_A * 2 + 0.5],
                                       title="Tipo de interés fijo — sin variación")
                st.plotly_chart(fig_fija, use_container_width=True)
            else:
                mat = np.array(eur_matrix)
                if len(mat) > 0:
                    p10, p50, p90 = np.percentile(mat, [10, 50, 90], axis=0)
                    x_ax = np.arange(1, len(p50) + 1)
                    fig_eur = go.Figure()
                    fig_eur.add_trace(go.Scatter(x=x_ax, y=p90, mode='lines',
                                                 line=dict(width=0), showlegend=False))
                    fig_eur.add_trace(go.Scatter(x=x_ax, y=p10, mode='lines',
                                                 line=dict(width=0), fill='tonexty',
                                                 fillcolor='rgba(0,100,250,0.15)', name='Rango'))
                    fig_eur.add_trace(go.Scatter(x=x_ax, y=p50, mode='lines',
                                                 line=dict(color='#0055aa', width=3), name='Mediana'))
                    fig_eur.update_layout(template='plotly_white', height=350,
                                          margin=dict(t=30), legend=dict(orientation="h", y=1.1))
                    st.plotly_chart(fig_eur, use_container_width=True)

        with c_e2:
            st.subheader("Cuota Mensual")
            fig2 = px.line(df_median_A, x='Mes', y='Cuota')
            fig2.update_traces(line_color='#d9534f', line_width=2.5)
            if es_autopromotor:
                fig2.add_vline(x=meses_carencia, line_dash="dot", annotation_text="Fin Carencia")
            st.plotly_chart(fig2, use_container_width=True)

    with tabs[1]:
        c_a1, c_a2 = st.columns(2)
        with c_a1:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df_base_median_A['Mes'],
                                      y=df_base_median_A['Intereses'].cumsum(),
                                      name='Base', line=dict(color='gray', dash='dash')))
            fig3.add_trace(go.Scatter(x=df_median_A['Mes'], y=df_median_A['Intereses'].cumsum(),
                                      name='Con Amort', line=dict(color='#d9534f', width=3)))
            st.plotly_chart(fig3, use_container_width=True)
        with c_a2:
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=df_base_median_A['Mes'], y=df_base_median_A['Saldo'],
                                      name='Base', line=dict(color='gray', dash='dash')))
            fig4.add_trace(go.Scatter(x=df_median_A['Mes'], y=df_median_A['Saldo'],
                                      fill='tozeroy', name='Real', line=dict(color='#5cb85c')))
            st.plotly_chart(fig4, use_container_width=True)

        # Botón de descarga también en modo individual
        st.download_button("📥 Descargar Excel", data=to_excel(df_median_A[cols_tabla]),
                           file_name='hipoteca.xlsx')

    with tabs[2]:
        fig5 = go.Figure()
        g_base = df_base_median_A['Cuota'] + coste_mes_seguros_A + total_gastos
        ah_base = ahorro_inicial + (ingresos - g_base).cumsum()
        pat_base = ah_base + (precio_vivienda - df_base_median_A['Saldo'])
        fig5.add_trace(go.Scatter(x=df_base_median_A['Mes'], y=pat_base,
                                  name='Base', line=dict(color='gray', dash='dot')))
        fig5.add_trace(go.Scatter(x=df_median_A['Mes'], y=df_median_A['Patrimonio'],
                                  name='Actual', line=dict(color='#6f42c1', width=3)))
        st.plotly_chart(fig5, use_container_width=True)

    with tabs[3]:
        st.subheader("Resumen Anual")
        anual_a = agregar_por_anio(df_median_A)
        st.download_button("📥 Descargar Excel Anual", data=to_excel(anual_a), file_name='anual.xlsx')
        st.dataframe(anual_a.style.format(
            {'Cuota_Media': '{:,.0f}', 'Intereses': '{:,.0f}', 'Capital_Amortizado': '{:,.0f}',
             'Saldo_Final': '{:,.0f}', 'Tasa': '{:.2f}%', 'Seguros': '{:,.0f}',
             'Gastos_Fijos': '{:,.0f}', 'Amort_Extra': '{:,.0f}'}),
            use_container_width=True)

    with tabs[4]:
        st.subheader("Riesgo")
        if n_sims < 10:
            st.warning("Selecciona 'Monte Carlo' con +50 simulaciones para ver el riesgo.")
        else:
            p5, p95 = np.percentile(kpis_int_A, [5, 95])
            fig_h = px.histogram(x=kpis_int_A, nbins=30, labels={'x': 'Coste Total'},
                                 color_discrete_sequence=['#8884d8'])
            fig_h.add_vline(x=p5, line_dash="dash", line_color="green", annotation_text="Mejor")
            fig_h.add_vline(x=p95, line_dash="dash", line_color="red", annotation_text="Peor")
            st.plotly_chart(fig_h, use_container_width=True)

    with tabs[5]:
        mostrar_liquidez(df_median_A, None, comparar=False)
