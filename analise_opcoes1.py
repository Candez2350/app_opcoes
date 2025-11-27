import streamlit as st
from yahooquery import Ticker
import pandas as pd
import ta
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# ================= CONFIGURAÇÃO =================
st.set_page_config(page_title="Radar Opções Master (Suportes Mistos)", page_icon="🦅", layout="wide")

# Inicialização do Session State
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
    if df.empty: return None
    df = df.reset_index()
    if 'date' in df.columns:
        df = df.set_index('date')
    
    cols_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
    df = df.rename(columns=cols_map)
    
    cols_price = ['Open', 'High', 'Low', 'Close']
    for c in cols_price:
        if c in df.columns:
            df[c] = df[c].replace(0, np.nan)
    df = df.dropna(subset=['Close'])
    
    mask_nan = df[['Open', 'High', 'Low']].isna().any(axis=1)
    if mask_nan.any():
        df.loc[mask_nan, 'Open'] = df.loc[mask_nan, 'Close']
        df.loc[mask_nan, 'High'] = df.loc[mask_nan, 'Close']
        df.loc[mask_nan, 'Low'] = df.loc[mask_nan, 'Close']

    return df

@st.cache_data(ttl=1800)
def obter_dados_multi_timeframe(ticker):
    if not ticker.endswith(".SA"): ticker += ".SA"
    try:
        t = Ticker(ticker)
        df_d = tratar_dataframe(t.history(period='1y', interval='1d'))
        df_w = tratar_dataframe(t.history(period='2y', interval='1wk'))
        df_h = tratar_dataframe(t.history(period='60d', interval='60m'))
        
        df_120 = None
        if df_h is not None and not df_h.empty:
            agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            df_120 = df_h.resample('2h').agg(agg_dict).dropna()

        if df_d is None or len(df_d) < 50: return None
        return {"D": df_d, "W": df_w, "120": df_120}
    except: return None

def encontrar_pivos(df, window=5):
    """Retorna pivôs locais (High/Low de curto prazo)."""
    df['Min_Local'] = df['Low'].rolling(window=window, center=True).min()
    df['Max_Local'] = df['High'].rolling(window=window, center=True).max()
    
    pivots_low = df[df['Low'] == df['Min_Local']]['Low']
    pivots_high = df[df['High'] == df['Max_Local']]['High']
    return pivots_low, pivots_high

def calcular_indicadores_tecnicos(df):
    if df is None or len(df) < 20: return None
    close = df['Close']
    
    df['EMA21'] = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    df['SMA50'] = ta.trend.SMAIndicator(close, window=50).sma_indicator()
    df['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df['MACD'] = ta.trend.MACD(close).macd()
    df['MACD_Signal'] = ta.trend.MACD(close).macd_signal()
    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], close, window=14).adx()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], close, window=14).average_true_range()
    
    df['Log_Ret'] = np.log(close / close.shift(1))
    df['HV20'] = df['Log_Ret'].rolling(window=20).std() * np.sqrt(252) * 100
    df['HV50'] = df['Log_Ret'].rolling(window=50).std() * np.sqrt(252) * 100
    return df

def analisar_timeframe_individual(df, periodo_nome):
    if df is None: return {"Viés": "N/A", "Score": 0}
    
    last = df.iloc[-1]
    preco_atual = last['Close']
    
    vies = "NEUTRO"
    score = 0
    motivos = []
    
    # 1. Estrutura de Médias
    if (last['Close'] > last['EMA21']) and (last['EMA21'] > last['SMA50']):
        vies = "ALTA"
        score += 2
        motivos.append("Médias Alta")
    elif (last['Close'] < last['EMA21']) and (last['EMA21'] < last['SMA50']):
        vies = "BAIXA"
        score += 2
        motivos.append("Médias Baixa")
    else:
        motivos.append("Lateral")

    # 2. Momentum
    if vies == "ALTA" and last['MACD'] > last['MACD_Signal']:
        score += 1
        motivos.append("MACD Compra")
    elif vies == "BAIXA" and last['MACD'] < last['MACD_Signal']:
        score += 1
        motivos.append("MACD Venda")

    # 3. PRICE ACTION DUAL (Intermediário vs Forte)
    pivots_low, pivots_high = encontrar_pivos(df, window=5)
    
    # -- SUPORTES --
    # Suporte Imediato (Pivô mais próximo abaixo)
    recent_lows = pivots_low.tail(30).values 
    suportes_abaixo = [p for p in recent_lows if p < preco_atual]
    sup_imediato = max(suportes_abaixo) if suportes_abaixo else (preco_atual * 0.9)
    
    # Suporte Forte (Mínima absoluta dos últimos 6 meses/120 candles)
    sup_forte = df['Low'].tail(120).min()
    
    # Se o suporte imediato for o mesmo que o forte (ou muito perto), consideramos um só
    if abs(sup_imediato - sup_forte) / sup_forte < 0.01:
        sup_imediato = sup_forte

    # -- RESISTÊNCIAS --
    # Resistência Imediata (Pivô mais próximo acima)
    recent_highs = pivots_high.tail(30).values
    resistencias_acima = [p for p in recent_highs if p > preco_atual]
    res_imediata = min(resistencias_acima) if resistencias_acima else (preco_atual * 1.1)
    
    # Resistência Forte (Máxima absoluta dos últimos 6 meses)
    res_forte = df['High'].tail(120).max()
    
    if abs(res_imediata - res_forte) / res_forte < 0.01:
        res_imediata = res_forte

    # Análise de Rompimento
    pa_status = "Dentro do Canal"
    if vies == "ALTA":
        if preco_atual > res_imediata: pa_status = "Rompimento Intermediário ⚠️"
        if preco_atual > res_forte: pa_status = "ROMPIMENTO MÁXIMA 🔥"
    elif vies == "BAIXA":
        if preco_atual < sup_imediato: pa_status = "Perda Intermediária ⚠️"
        if preco_atual < sup_forte: pa_status = "PERDA MÍNIMA (Crash) 🩸"
        
    return {
        "Viés": vies,
        "Score": score,
        "RSI": last['RSI'],
        "PA_Status": pa_status,
        "Sup_Imediato": sup_imediato,
        "Sup_Forte": sup_forte,
        "Res_Imediata": res_imediata,
        "Res_Forte": res_forte,
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
    obs_final = []

    # Confluência
    if analise_d['Viés'] == "ALTA":
        if analise_w['Viés'] in ["ALTA", "NEUTRO"]: 
            score_final += 3
            decisao_final = "ALTA"
            if analise_w['Viés'] == "ALTA": score_final += 1; obs_final.append("Semanal ✅")
            if analise_120['Viés'] == "ALTA": score_final += 1; obs_final.append("Intraday ✅")
            if "ROMPIMENTO" in analise_d['PA_Status']: score_final += 1; obs_final.append("Rompimento 🔥")
    
    elif analise_d['Viés'] == "BAIXA":
        if analise_w['Viés'] in ["BAIXA", "NEUTRO"]:
            score_final += 3
            decisao_final = "BAIXA"
            if analise_w['Viés'] == "BAIXA": score_final += 1; obs_final.append("Semanal ✅")
            if analise_120['Viés'] == "BAIXA": score_final += 1; obs_final.append("Intraday ✅")
            if "PERDA" in analise_d['PA_Status']: score_final += 1; obs_final.append("Perda Suporte 🩸")

    # Volatilidade e Setup
    vol_status = "NORMAL"
    cond_vol = "media"
    if last_d['HV20'] < last_d['HV50'] * 0.9: vol_status, cond_vol = "📉 Barata", "baixa"
    elif last_d['HV20'] > last_d['HV50'] * 1.2: vol_status, cond_vol = "📈 Cara", "alta"

    setup_sugerido = "AGUARDAR"
    if decisao_final != "NEUTRO" and score_final >= 4:
        if last_d['ADX'] > 25 and cond_vol != "alta": setup_sugerido = "COMPRA A SECO"
        elif cond_vol == "alta": setup_sugerido = "TRAVA"
        else: setup_sugerido = "TRAVA OU SECO"

    # Alvos/Stop
    atr = last_d['ATR']
    if decisao_final == "ALTA":
        stop = analise_d['Sup_Imediato'] # Stop no suporte local
        alvo = last_d['Close'] + (3*atr)
    elif decisao_final == "BAIXA":
        stop = analise_d['Res_Imediata']
        alvo = last_d['Close'] - (3*atr)
    else: stop, alvo = 0, 0

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
        "analise_w": analise_w, "analise_d": analise_d, "analise_120": analise_120, "df_chart": df_d
    }

def criar_grafico_mtf(df, ticker, analise_d):
    fig = go.Figure()

    # Candles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Preço Daily'
    ))

    # Médias
    if 'EMA21' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], mode='lines', name='EMA21', line=dict(color='cyan', width=1)))
    if 'SMA50' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA50', line=dict(color='yellow', width=1)))
        
    # === PRICE ACTION VISUAL ===
    s_imediato = analise_d.get('Sup_Imediato', 0)
    s_forte = analise_d.get('Sup_Forte', 0)
    r_imediata = analise_d.get('Res_Imediata', 0)
    r_forte = analise_d.get('Res_Forte', 0)
    
    # 1. Resistência Forte (Vermelho Escuro, Contínuo)
    fig.add_shape(type="line", x0=df.index[-120], y0=r_forte, x1=df.index[-1], y1=r_forte, 
                  line=dict(color="#B71C1C", width=2, dash="solid"))
    fig.add_annotation(x=df.index[-1], y=r_forte, text=f"Topo Forte: {r_forte:.2f}", showarrow=False, yshift=10, font=dict(color="#B71C1C"))

    # 2. Resistência Imediata (Vermelho Claro, Pontilhado)
    if r_imediata < r_forte: # Só mostra se for diferente
        fig.add_shape(type="line", x0=df.index[-30], y0=r_imediata, x1=df.index[-1], y1=r_imediata, 
                      line=dict(color="#EF5350", width=1, dash="dash"))
        fig.add_annotation(x=df.index[-5], y=r_imediata, text=f"Res. Local: {r_imediata:.2f}", showarrow=False, yshift=10, font=dict(color="#EF5350"))

    # 3. Suporte Imediato (Verde Claro, Pontilhado) - Ex: Os 32,00 da PETR4
    if s_imediato > s_forte:
        fig.add_shape(type="line", x0=df.index[-30], y0=s_imediato, x1=df.index[-1], y1=s_imediato, 
                      line=dict(color="#66BB6A", width=1, dash="dash"))
        fig.add_annotation(x=df.index[-5], y=s_imediato, text=f"Sup. Local: {s_imediato:.2f}", showarrow=False, yshift=-10, font=dict(color="#66BB6A"))

    # 4. Suporte Forte (Verde Escuro, Contínuo) - Ex: Os 29,55 da PETR4
    fig.add_shape(type="line", x0=df.index[-120], y0=s_forte, x1=df.index[-1], y1=s_forte, 
                  line=dict(color="#1B5E20", width=2, dash="solid"))
    fig.add_annotation(x=df.index[-1], y=s_forte, text=f"Fundo Forte: {s_forte:.2f}", showarrow=False, yshift=-10, font=dict(color="#1B5E20"))
    
    fig.update_layout(title=f"Análise Técnica Daily: {ticker}", template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=50, r=50, t=50, b=50))
    return fig

# ================= INTERFACE =================
st.title("🦅 Radar Opções Pro: Dual Support System")
st.markdown("""
**Análise de Price Action Avançada:**
* **Linha Contínua (Escura):** Suportes e Resistências **Fortes** (Estruturais).
* **Linha Pontilhada (Clara):** Suportes e Resistências **Imediatos** (Pivôs).
""")

# Sidebar
selecao = IBXX_FULL_LIST
if not st.sidebar.checkbox("Analisar Lista Completa", value=False):
    selecao = st.sidebar.multiselect("Carteira:", IBXX_FULL_LIST, default=["PETR4", "VALE3", "BOVA11", "MGLU3", "PRIO3", "BBAS3"])

if st.sidebar.button("🔍 Iniciar Varredura"):
    resultados = []
    progresso = st.progress(0)
    status = st.empty()
    
    for i, ticker in enumerate(selecao):
        status.text(f"Lendo {ticker}...")
        dados = obter_dados_multi_timeframe(ticker)
        if dados:
            res = analisar_ativo_completo(ticker, dados)
            if res: resultados.append(res)
        progresso.progress((i + 1) / len(selecao))
    
    status.empty(); progresso.empty()
    st.session_state.dados_analise = resultados
    st.session_state.analise_realizada = True

# ================= EXIBIÇÃO =================
if st.session_state.analise_realizada:
    df_res = pd.DataFrame(st.session_state.dados_analise)
    
    if not df_res.empty:
        df_alta = df_res[df_res['Direção'] == "ALTA"].sort_values('Score', ascending=False)
        df_baixa = df_res[df_res['Direção'] == "BAIXA"].sort_values('Score', ascending=False)
        
        c1, c2 = st.columns(2)
        with c1: 
            st.success(f"🚀 CALL / ALTA: {len(df_alta)}")
            if not df_alta.empty: st.dataframe(df_alta[['Ativo', 'Preço', 'Score', 'Setup', 'Stop']], use_container_width=True, hide_index=True)
        with c2: 
            st.error(f"🩸 PUT / BAIXA: {len(df_baixa)}")
            if not df_baixa.empty: st.dataframe(df_baixa[['Ativo', 'Preço', 'Score', 'Setup', 'Stop']], use_container_width=True, hide_index=True)
        
        st.divider()
        st.subheader("🕵️‍♂️ Detalhamento Técnico")
        ativos = df_res['Ativo'].tolist()
        escolha = st.selectbox("Selecione Ativo:", ativos)
        
        if escolha:
            d_ativo = next(i for i in st.session_state.dados_analise if i["Ativo"] == escolha)
            analise_d = d_ativo['analise_d']
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Preço", f"R$ {d_ativo['Preço']:.2f}")
            c2.metric("Sup. Imediato", f"R$ {analise_d['Sup_Imediato']:.2f}")
            c3.metric("Sup. Forte (Fundo)", f"R$ {analise_d['Sup_Forte']:.2f}", delta_color="normal")
            c4.metric("Alvo", f"R$ {d_ativo['Alvo']:.2f}")
            
            st.plotly_chart(criar_grafico_mtf(d_ativo['df_chart'], escolha, analise_d), use_container_width=True)
