import streamlit as st
from yahooquery import Ticker
import pandas as pd
import ta
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# ================= CONFIGURAÇÃO =================
st.set_page_config(page_title="Radar Opções Master (MTF & Price Action)", page_icon="🦅", layout="wide")

# Inicialização do Session State (Memória)
if 'dados_analise' not in st.session_state:
    st.session_state.dados_analise = []
if 'analise_realizada' not in st.session_state:
    st.session_state.analise_realizada = False

# ================= LISTA IBrX 100 =================
IBXX_FULL_LIST = [
    "ABEV3", "ALOS3", "ALPA4", "ASAI3", "AZUL4", "B3SA3", "BBAS3", "BBDC3", "BBDC4",
    "BBSE3", "BEEF3", "BPAC11", "BPAN4", "BRAP4", "BRAV3", "BRFS3", "BRKM5", "CAML3",
    "CASH3", "CCRO3", "CMIG4", "CMIN3", "COGN3", "CPFE3", "CPLE6", "CRFB3", "CSAN3",
    "CSNA3", "CVCB3", "CYRE3", "DXCO3", "ECOR3", "EGIE3", "ELET3", "ELET6", "EMBR3",
    "ENEV3", "ENGI11", "EQTL3", "EZTC3", "FLRY3", "GGBR4", "GOAU4", "GOLL4", "HAPV3",
    "HYPE3", "IGTI11", "IRBR3", "ITSA4", "ITUB4", "JBSS3", "JHSF3", "KLBN11", "LREN3",
    "LWSA3", "MGLU3", "MOVI3", "MRFG3", "MRVE3", "MULT3", "NTCO3", "PCAR3", "PETR3",
    "PETR4", "PETZ3", "PLPL3", "POMO4", "PRIO3", "PSSA3", "RADL3", "RAIL3", "RAIZ4",
    "RANI3", "RDOR3", "RECV3", "RENT3", "ROMI3", "SANB11", "SBSP3", "SLCE3", "SMTO3",
    "STBP3", "SUZB3", "TAEE11", "TIMS3", "TOTS3", "TRPL4", "UGPA3", "USIM5", "VALE3",
    "VBBR3", "VIVA3", "VIVT3", "WEGE3", "YDUQ3", "VAMO3", "BOVA11", "SMAL11"
]
IBXX_FULL_LIST.sort()

# ================= CÁLCULOS E DADOS =================

def tratar_dataframe(df):
    """Limpa e padroniza o DataFrame vindo do yahooquery."""
    if df.empty: return None
    df = df.reset_index()
    if 'date' in df.columns:
        df = df.set_index('date')
    
    # Padronização de nomes
    cols_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
    df = df.rename(columns=cols_map)
    
    # Tratamento de zeros (Bug Yahoo)
    cols_price = ['Open', 'High', 'Low', 'Close']
    for c in cols_price:
        if c in df.columns:
            df[c] = df[c].replace(0, np.nan)
    df = df.dropna(subset=['Close']) # Remove dias sem close
    
    # Preenchimento de High/Low/Open corrompidos
    mask_nan = df[['Open', 'High', 'Low']].isna().any(axis=1)
    if mask_nan.any():
        df.loc[mask_nan, 'Open'] = df.loc[mask_nan, 'Close']
        df.loc[mask_nan, 'High'] = df.loc[mask_nan, 'Close']
        df.loc[mask_nan, 'Low'] = df.loc[mask_nan, 'Close']

    return df

@st.cache_data(ttl=1800) # Cache de 30 min
def obter_dados_multi_timeframe(ticker):
    if not ticker.endswith(".SA"): ticker += ".SA"
    
    try:
        t = Ticker(ticker)
        
        # 1. Dados Diários (Principal) - Pega 1 ano para médias longas
        df_d = t.history(period='1y', interval='1d')
        df_d = tratar_dataframe(df_d)
        
        # 2. Dados Semanais (Tendência Macro)
        df_w = t.history(period='2y', interval='1wk')
        df_w = tratar_dataframe(df_w)

        # 3. Dados Intraday (Para gerar 120min) - Yahoo limita intraday a 60dias aprox
        df_h = t.history(period='60d', interval='60m')
        df_h = tratar_dataframe(df_h)
        
        # Resample de 60m para 120m (2H)
        df_120 = None
        if df_h is not None and not df_h.empty:
            agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            df_120 = df_h.resample('2h').agg(agg_dict).dropna()

        if df_d is None or len(df_d) < 50: return None
        
        return {"D": df_d, "W": df_w, "120": df_120}
        
    except Exception as e:
        return None

def calcular_indicadores_tecnicos(df):
    if df is None or len(df) < 20: return None
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # Médias Móveis
    df['EMA21'] = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    df['SMA50'] = ta.trend.SMAIndicator(close, window=50).sma_indicator()
    
    # Osciladores
    df['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df['MACD'] = ta.trend.MACD(close).macd()
    df['MACD_Signal'] = ta.trend.MACD(close).macd_signal()
    df['ADX'] = ta.trend.ADXIndicator(high, low, close, window=14).adx()
    df['ATR'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    
    # === PRICE ACTION (SUPORTE E RESISTÊNCIA) ===
    # Canal de Donchian de 20 períodos
    df['Resistencia_20'] = high.rolling(window=20).max().shift(1)
    df['Suporte_20'] = low.rolling(window=20).min().shift(1)
    
    # Volatilidade Histórica
    df['Log_Ret'] = np.log(close / close.shift(1))
    df['HV20'] = df['Log_Ret'].rolling(window=20).std() * np.sqrt(252) * 100
    df['HV50'] = df['Log_Ret'].rolling(window=50).std() * np.sqrt(252) * 100
    
    return df

def analisar_timeframe_individual(df, periodo_nome):
    if df is None: return {"Viés": "N/A", "Score": 0, "Detalhe": "Dados Insuficientes"}
    
    last = df.iloc[-1]
    
    vies = "NEUTRO"
    score = 0
    motivos = []
    
    # 1. Estrutura de Médias
    if (last['Close'] > last['EMA21']) and (last['EMA21'] > last['SMA50']):
        vies = "ALTA"
        score += 2
        motivos.append("Médias Alinhadas")
    elif (last['Close'] < last['EMA21']) and (last['EMA21'] < last['SMA50']):
        vies = "BAIXA"
        score += 2
        motivos.append("Médias Alinhadas")
    else:
        motivos.append("Lateral")

    # 2. Momentum (MACD)
    if vies == "ALTA" and last['MACD'] > last['MACD_Signal']:
        score += 1
        motivos.append("MACD Compra")
    elif vies == "BAIXA" and last['MACD'] < last['MACD_Signal']:
        score += 1
        motivos.append("MACD Venda")

    # 3. Price Action Simples
    dist_resistencia = (last['Resistencia_20'] - last['Close']) / last['Close']
    dist_suporte = (last['Close'] - last['Suporte_20']) / last['Close']
    
    pa_status = ""
    if vies == "ALTA":
        if last['Close'] > last['Resistencia_20']: pa_status = "ROMPIMENTO DE TOPO 🚀"
        elif dist_resistencia < 0.02: pa_status = "Testando Resistência"
        else: pa_status = "Dentro do Canal"
    elif vies == "BAIXA":
        if last['Close'] < last['Suporte_20']: pa_status = "PERDA DE FUNDO 📉"
        elif dist_suporte < 0.02: pa_status = "Testando Suporte"
        else: pa_status = "Dentro do Canal"
        
    return {
        "Viés": vies,
        "Score": score,
        "RSI": last['RSI'],
        "PA_Status": pa_status,
        "Suporte": last['Suporte_20'],
        "Resistencia": last['Resistencia_20'],
        "Motivos": ", ".join(motivos)
    }

def analisar_ativo_completo(ticker, dados_dict):
    df_d = calcular_indicadores_tecnicos(dados_dict["D"])
    df_w = calcular_indicadores_tecnicos(dados_dict["W"])
    df_120 = calcular_indicadores_tecnicos(dados_dict["120"])
    
    if df_d is None: return None

    analise_w = analisar_timeframe_individual(df_w, "Semanal")
    analise_d = analisar_timeframe_individual(df_d, "Diário")
    analise_120 = analisar_timeframe_individual(df_120, "120min")
    
    last_d = df_d.iloc[-1]
    
    score_final = 0
    decisao_final = "NEUTRO"
    setup_sugerido = "-"
    obs_final = []

    # O Diário é o Mandante (Gatilho)
    # O Semanal é o Filtro (Permissão)
    
    if analise_d['Viés'] == "ALTA":
        if analise_w['Viés'] in ["ALTA", "NEUTRO"]: 
            score_final += 3
            decisao_final = "ALTA"
            
            if analise_w['Viés'] == "ALTA": 
                score_final += 1
                obs_final.append("Semanal ✅")
            if analise_120['Viés'] == "ALTA":
                score_final += 1
                obs_final.append("Intraday ✅")
            if "ROMPIMENTO" in analise_d['PA_Status']:
                score_final += 1
                obs_final.append("Breakout 🔥")
    
    elif analise_d['Viés'] == "BAIXA":
        if analise_w['Viés'] in ["BAIXA", "NEUTRO"]:
            score_final += 3
            decisao_final = "BAIXA"
            
            if analise_w['Viés'] == "BAIXA": 
                score_final += 1
                obs_final.append("Semanal ✅")
            if analise_120['Viés'] == "BAIXA":
                score_final += 1
                obs_final.append("Intraday ✅")
            if "PERDA" in analise_d['PA_Status']:
                score_final += 1
                obs_final.append("Breakdown 🩸")

    # Volatilidade
    vol_status = "NORMAL"
    cond_vol = "media"
    if last_d['HV20'] < last_d['HV50'] * 0.9:
        vol_status = "📉 Barata"
        cond_vol = "baixa"
    elif last_d['HV20'] > last_d['HV50'] * 1.2:
        vol_status = "📈 Cara"
        cond_vol = "alta"

    # Setup
    if decisao_final != "NEUTRO" and score_final >= 4:
        if last_d['ADX'] > 25 and cond_vol != "alta":
            setup_sugerido = "COMPRA A SECO"
        elif cond_vol == "alta":
            setup_sugerido = "TRAVA"
        else:
            setup_sugerido = "TRAVA OU SECO"
    else:
        setup_sugerido = "AGUARDAR"

    # Alvos/Stop
    atr = last_d['ATR']
    if decisao_final == "ALTA":
        stop = last_d['Close'] - (2 * atr)
        alvo = last_d['Close'] + (3 * atr)
    elif decisao_final == "BAIXA":
        stop = last_d['Close'] + (2 * atr)
        alvo = last_d['Close'] - (3 * atr)
    else:
        stop, alvo = 0, 0

    return {
        "Ativo": ticker,
        "Preço": last_d['Close'],
        "Direção": decisao_final,
        "Score": score_final,
        "Setup": setup_sugerido,
        "Volatilidade": vol_status,
        "HV20": f"{last_d['HV20']:.1f}%",
        "Stop": stop,
        "Alvo": alvo,
        "Observacoes": ", ".join(obs_final) if obs_final else "Aguardando",
        "analise_w": analise_w,
        "analise_d": analise_d,
        "analise_120": analise_120,
        "df_chart": df_d
    }

def criar_grafico_mtf(df, ticker, analise_d):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Preço Daily'
    ))

    if 'EMA21' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], mode='lines', name='EMA21', line=dict(color='cyan', width=1)))
    if 'SMA50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA50', line=dict(color='yellow', width=1)))
        
    sup = analise_d.get('Suporte', 0)
    res = analise_d.get('Resistencia', 0)
    
    # Linhas de PA
    fig.add_shape(type="line", x0=df.index[-30], y0=res, x1=df.index[-1], y1=res, 
                  line=dict(color="Red", width=1, dash="dash"), name="Resistência")
    fig.add_shape(type="line", x0=df.index[-30], y0=sup, x1=df.index[-1], y1=sup, 
                  line=dict(color="Green", width=1, dash="dash"), name="Suporte")
    
    fig.update_layout(
        title=f"Gráfico Diário + Price Action: {ticker}",
        template="plotly_dark",
        height=500,
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig

# ================= INTERFACE =================
st.title("🦅 Radar Opções Pro: MTF & Price Action")
st.markdown("""
**Metodologia Triple Screen Adaptada:**
* **Semanal (Segurança):** Tendência Macro.
* **Diário (Gatilho):** Setup + Suporte/Resistência.
* **120 Min (Timing):** Refino intraday.
""")

# Sidebar
selecao = IBXX_FULL_LIST
if not st.sidebar.checkbox("Analisar Lista Completa (Pode demorar)", value=False):
    selecao = st.sidebar.multiselect("Carteira Personalizada:", IBXX_FULL_LIST, default=["PETR4", "VALE3", "BOVA11", "MGLU3", "PRIO3", "BBAS3"])

# Botão de Execução
if st.sidebar.button("🔍 Iniciar Varredura"):
    resultados = []
    
    progresso = st.progress(0)
    status_txt = st.empty()
    
    for i, ticker in enumerate(selecao):
        status_txt.text(f"Processando {ticker}...")
        
        dados = obter_dados_multi_timeframe(ticker)
        if dados:
            res = analisar_ativo_completo(ticker, dados)
            if res:
                resultados.append(res)
        
        progresso.progress((i + 1) / len(selecao))
    
    status_txt.empty()
    progresso.empty()
    
    # SALVA NA MEMÓRIA (SESSION STATE)
    st.session_state.dados_analise = resultados
    st.session_state.analise_realizada = True

# ================= EXIBIÇÃO DOS RESULTADOS (FORA DO IF DO BOTÃO) =================
if st.session_state.analise_realizada:
    
    df_res = pd.DataFrame(st.session_state.dados_analise)
    
    if not df_res.empty:
        df_alta = df_res[df_res['Direção'] == "ALTA"].sort_values('Score', ascending=False)
        df_baixa = df_res[df_res['Direção'] == "BAIXA"].sort_values('Score', ascending=False)
        df_neutro = df_res[df_res['Direção'] == "NEUTRO"].sort_values('Score', ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"🚀 CALL / ALTA: {len(df_alta)}")
            if not df_alta.empty:
                st.dataframe(df_alta[['Ativo', 'Preço', 'Score', 'Setup', 'Volatilidade', 'Observacoes']], use_container_width=True, hide_index=True)
        
        with col2:
            st.error(f"🩸 PUT / BAIXA: {len(df_baixa)}")
            if not df_baixa.empty:
                st.dataframe(df_baixa[['Ativo', 'Preço', 'Score', 'Setup', 'Volatilidade', 'Observacoes']], use_container_width=True, hide_index=True)
        
        with st.expander(f"⚠️ Neutros / Observação ({len(df_neutro)})"):
            if not df_neutro.empty:
                st.dataframe(df_neutro[['Ativo', 'Preço', 'Volatilidade', 'Observacoes']], use_container_width=True, hide_index=True)
        
        # === ÁREA DE DETALHES (PERSISTENTE) ===
        st.divider()
        st.subheader("🕵️‍♂️ Detalhamento Técnico")
        
        ativos_disponiveis = df_res['Ativo'].tolist()
        ativo_escolhido = st.selectbox("Selecione um ativo para Raio-X completo:", ativos_disponiveis)
        
        if ativo_escolhido:
            # Busca o ativo na memória
            data_ativo = next(item for item in st.session_state.dados_analise if item["Ativo"] == ativo_escolhido)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Preço", f"R$ {data_ativo['Preço']:.2f}")
            c2.metric("Score", f"{data_ativo['Score']}/6")
            c3.metric("Stop", f"R$ {data_ativo['Stop']:.2f}")
            c4.metric("Alvo", f"R$ {data_ativo['Alvo']:.2f}")
            
            st.markdown("#### ⏳ Comparativo Multi-Timeframe")
            w = data_ativo['analise_w']
            d = data_ativo['analise_d']
            h = data_ativo['analise_120']
            
            cols = st.columns(3)
            with cols[0]:
                st.info("📅 SEMANAL")
                st.write(f"Viés: **{w['Viés']}**")
                st.write(f"RSI: {w['RSI']:.1f}")
                st.caption(w['PA_Status'])
            with cols[1]:
                st.warning("📆 DIÁRIO")
                st.write(f"Viés: **{d['Viés']}**")
                st.write(f"RSI: {d['RSI']:.1f}")
                st.write(f"Resistência: {d['Resistencia']:.2f}")
                st.write(f"Suporte: {d['Suporte']:.2f}")
                st.caption(d['PA_Status'])
            with cols[2]:
                st.success("⏱️ 120 MIN")
                st.write(f"Viés: **{h['Viés']}**")
                st.write(f"RSI: {h['RSI']:.1f}")
                st.caption(h['PA_Status'])
            
            st.plotly_chart(criar_grafico_mtf(data_ativo['df_chart'], ativo_escolhido, d), use_container_width=True)

    else:
        st.warning("Nenhum dado retornado.")
