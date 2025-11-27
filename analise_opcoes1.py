import streamlit as st
from yahooquery import Ticker
import pandas as pd
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# ================= CONFIGURAÇÃO VISUAL =================
st.set_page_config(
    page_title="Vector 3 | Algorithmic Scanner", 
    page_icon="💠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    if 'date' in df.columns: df = df.set_index('date')
    cols_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
    df = df.rename(columns=cols_map)
    cols_price = ['Open', 'High', 'Low', 'Close']
    for c in cols_price:
        if c in df.columns: df[c] = df[c].replace(0, np.nan)
    df = df.dropna(subset=['Close'])
    
    # Tratamento de zeros
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
    
    # Pre-calculo para backtest (Resistência e Suporte Históricos)
    df['Resistencia_Hist'] = df['High'].rolling(window=20).max().shift(1)
    df['Suporte_Hist'] = df['Low'].rolling(window=20).min().shift(1)
    
    return df

def analisar_timeframe_individual(df, periodo_nome):
    if df is None: return {"Viés": "N/A", "Score": 0, "Detalhe": "N/A"}
    last = df.iloc[-1]
    preco_atual = last['Close']
    vies = "NEUTRO"
    score = 0
    motivos = []
    
    if (last['Close'] > last['EMA21']) and (last['EMA21'] > last['SMA50']):
        vies = "ALTA"; score += 2; motivos.append("Tendência Alta")
    elif (last['Close'] < last['EMA21']) and (last['EMA21'] < last['SMA50']):
        vies = "BAIXA"; score += 2; motivos.append("Tendência Baixa")
    else: motivos.append("Lateral")

    macd_ok = False
    if vies == "ALTA" and last['MACD'] > last['MACD_Signal']:
        score += 1; macd_ok = True; motivos.append("MACD Compra")
    elif vies == "BAIXA" and last['MACD'] < last['MACD_Signal']:
        score += 1; macd_ok = True; motivos.append("MACD Venda")

    pivots_low, pivots_high = encontrar_pivos(df, window=5)
    
    # Suportes
    recent_lows = pivots_low.tail(30).values 
    suportes_abaixo = [p for p in recent_lows if p < preco_atual]
    sup_imediato = max(suportes_abaixo) if suportes_abaixo else (preco_atual * 0.9)
    sup_forte = df['Low'].tail(120).min()
    if abs(sup_imediato - sup_forte) / sup_forte < 0.01: sup_imediato = sup_forte

    # Resistências
    recent_highs = pivots_high.tail(30).values
    resistencias_acima = [p for p in recent_highs if p > preco_atual]
    res_imediata = min(resistencias_acima) if resistencias_acima else (preco_atual * 1.1)
    res_forte = df['High'].tail(120).max()
    if abs(res_imediata - res_forte) / res_forte < 0.01: res_imediata = res_forte

    pa_status = "Dentro do Canal"
    rompimento, perda = False, False
    
    if vies == "ALTA":
        if preco_atual > res_imediata: pa_status, rompimento = "Rompimento Intermediário ⚠️", True
        if preco_atual > res_forte: pa_status, rompimento = "ROMPIMENTO MÁXIMA 🔥", True
    elif vies == "BAIXA":
        if preco_atual < sup_imediato: pa_status, perda = "Perda Intermediária ⚠️", True
        if preco_atual < sup_forte: pa_status, perda = "PERDA MÍNIMA (Crash) 🩸", True
        
    return {
        "Viés": vies, "Score": score, "MACD_OK": macd_ok, "RSI": last['RSI'],
        "PA_Status": pa_status, "Rompimento": rompimento, "Perda": perda,
        "Sup_Imediato": sup_imediato, "Sup_Forte": sup_forte,
        "Res_Imediata": res_imediata, "Res_Forte": res_forte,
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
    breakdown = {"Tendência Diária": 0, "MACD Diário": 0, "Confluência Semanal": 0, "Confluência Intraday": 0, "Price Action": 0}

    if analise_d['Viés'] == "ALTA":
        score_final += 2; breakdown["Tendência Diária"] = 2
        if analise_d['MACD_OK']: score_final += 1; breakdown["MACD Diário"] = 1
        if analise_w['Viés'] in ["ALTA", "NEUTRO"]: 
            decisao_final = "ALTA"
            if analise_w['Viés'] == "ALTA": score_final += 1; breakdown["Confluência Semanal"] = 1; obs_final.append("Semanal ✅")
            if analise_120['Viés'] == "ALTA": score_final += 1; breakdown["Confluência Intraday"] = 1; obs_final.append("Intraday ✅")
            if analise_d['Rompimento']: score_final += 1; breakdown["Price Action"] = 1; obs_final.append("Rompimento 🔥")
    
    elif analise_d['Viés'] == "BAIXA":
        score_final += 2; breakdown["Tendência Diária"] = 2
        if analise_d['MACD_OK']: score_final += 1; breakdown["MACD Diário"] = 1
        if analise_w['Viés'] in ["BAIXA", "NEUTRO"]:
            decisao_final = "BAIXA"
            if analise_w['Viés'] == "BAIXA": score_final += 1; breakdown["Confluência Semanal"] = 1; obs_final.append("Semanal ✅")
            if analise_120['Viés'] == "BAIXA": score_final += 1; breakdown["Confluência Intraday"] = 1; obs_final.append("Intraday ✅")
            if analise_d['Perda']: score_final += 1; breakdown["Price Action"] = 1; obs_final.append("Perda Suporte 🩸")

    vol_status = "NORMAL"
    cond_vol = "media"
    if last_d['HV20'] < last_d['HV50'] * 0.9: vol_status, cond_vol = "📉 Barata", "baixa"
    elif last_d['HV20'] > last_d['HV50'] * 1.2: vol_status, cond_vol = "📈 Cara", "alta"

    setup_sugerido = "AGUARDAR"
    if decisao_final != "NEUTRO" and score_final >= 4:
        if last_d['ADX'] > 25 and cond_vol != "alta": setup_sugerido = "COMPRA A SECO"
        elif cond_vol == "alta": setup_sugerido = "TRAVA"
        else: setup_sugerido = "TRAVA OU SECO"

    atr = last_d['ATR']
    stop_tecnico = 0.0
    alvo_tecnico = 0.0
    payoff = 0.0
    
    if decisao_final == "ALTA":
        stop_tecnico = last_d['Close'] - (1.5 * atr) 
        alvo_tecnico = last_d['Close'] + (3.0 * atr)
        risco = last_d['Close'] - stop_tecnico
        retorno = alvo_tecnico - last_d['Close']
        if risco > 0: payoff = retorno / risco
    elif decisao_final == "BAIXA":
        stop_tecnico = last_d['Close'] + (1.5 * atr)
        alvo_tecnico = last_d['Close'] - (3.0 * atr)
        risco = stop_tecnico - last_d['Close']
        retorno = last_d['Close'] - alvo_tecnico
        if risco > 0: payoff = retorno / risco

    return {
        "Ativo": ticker, "Preço": last_d['Close'], "Direção": decisao_final,
        "Score": score_final, "Breakdown": breakdown, "Setup": setup_sugerido,
        "Volatilidade": vol_status, "Stop_Tecnico": stop_tecnico, "Alvo": alvo_tecnico, "Payoff": payoff,
        "Observacoes": ", ".join(obs_final) if obs_final else "Aguardando",
        "analise_w": analise_w, "analise_d": analise_d, "analise_120": analise_120, 
        "df_chart_d": df_d, "df_chart_w": df_w, "df_chart_120": df_120
    }

def gerar_relatorio_textual(data):
    ticker = data['Ativo']
    direcao = data['Direção']
    score = data['Score']
    bk = data['Breakdown']
    w, d = data['analise_w'], data['analise_d']
    
    emoji = "🐂" if direcao == "ALTA" else ("🐻" if direcao == "BAIXA" else "⚖️")
    texto = f"### {emoji} Relatório Técnico: {ticker}\n\n"
    
    texto += "**1. Diagnóstico Geral:**\n"
    if direcao == "NEUTRO":
        texto += f"O ativo encontra-se em zona de indefinição. O Score atual é de **{score}/6**. "
        if d['Viés'] != w['Viés']: texto += f"Divergência: Diário (**{d['Viés']}**) vs Semanal (**{w['Viés']}**).\n\n"
        else: texto += "O ativo está lateral.\n\n"
    else:
        forca = "Forte" if score >= 5 else "Moderada"
        texto += f"O ativo apresenta tendência de **{direcao}** com força **{forca}** (Score {score}/6).\n\n"

    texto += "**2. Fundamentos da Tese (Pontos Positivos):**\n"
    if bk['Tendência Diária'] > 0: texto += "- ✅ **Tendência Diária:** Médias alinhadas a favor.\n"
    if bk['MACD Diário'] > 0: texto += "- ✅ **Momentum:** MACD confirmando o movimento.\n"
    if bk['Confluência Semanal'] > 0: texto += "- ✅ **Macro:** Gráfico Semanal alinhado.\n"
    if bk['Confluência Intraday'] > 0: texto += "- ✅ **Timing:** Intraday (120min) alinhado.\n"
    if bk['Price Action'] > 0: texto += f"- ✅ **Price Action:** Rompimento de {data['Direção'].lower()} detectado.\n"
    
    riscos = []
    if bk['Confluência Semanal'] == 0 and direcao != "NEUTRO": riscos.append("Semanal ainda não confirmou (divergência).")
    if bk['Price Action'] == 0 and direcao != "NEUTRO": riscos.append("Preço ainda em congestão (sem rompimento claro).")
    if bk['MACD Diário'] == 0 and direcao != "NEUTRO": riscos.append("MACD atrasado ou divergente.")
    
    if riscos:
        texto += "\n**3. Pontos de Atenção (Riscos):**\n"
        for r in riscos: texto += f"- ⚠️ {r}\n"
    
    texto += "\n**4. Níveis Chave (Price Action):**\n"
    if direcao == "ALTA":
        texto += f"- **Suporte (Stop):** Região de R\$ {d['Sup_Imediato']:.2f}\n"
        texto += f"- **Resistência (Alvo):** Região de R\$ {d['Res_Imediata']:.2f}\n"
        texto += "- **Cenário:** Caminho livre até a resistência caso mantenha o suporte."
    elif direcao == "BAIXA":
        texto += f"- **Resistência (Stop):** Região de R\$ {d['Res_Imediata']:.2f}\n"
        texto += f"- **Suporte (Alvo):** Região de R\$ {d['Sup_Imediato']:.2f}\n"
        texto += "- **Cenário:** Espaço para cair até o suporte caso não rompa a resistência."
    else:
        texto += f"- Ativo 'preso' entre **R\$ {d['Sup_Imediato']:.2f}** e **R\$ {d['Res_Imediata']:.2f}**."

    return texto

def criar_grafico_dinamico(df, ticker, analise_d, tipo_grafico):
    if df is None or df.empty:
        fig = go.Figure(); fig.update_layout(title=f"Dados insuficientes para {tipo_grafico}"); return fig

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=f'{ticker}'))
    if 'EMA21' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['EMA21'], mode='lines', name='EMA21', line=dict(color='cyan', width=1)))
    if 'SMA50' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA50', line=dict(color='yellow', width=1)))
        
    if tipo_grafico == "Diário":
        s_imediato, s_forte = analise_d.get('Sup_Imediato', 0), analise_d.get('Sup_Forte', 0)
        r_imediata, r_forte = analise_d.get('Res_Imediata', 0), analise_d.get('Res_Forte', 0)
        
        fig.add_shape(type="line", x0=df.index[-120], y0=r_forte, x1=df.index[-1], y1=r_forte, line=dict(color="#B71C1C", width=2, dash="solid"))
        if r_imediata < r_forte: fig.add_shape(type="line", x0=df.index[-30], y0=r_imediata, x1=df.index[-1], y1=r_imediata, line=dict(color="#EF5350", width=1, dash="dash"))
        if s_imediato > s_forte: fig.add_shape(type="line", x0=df.index[-30], y0=s_imediato, x1=df.index[-1], y1=s_imediato, line=dict(color="#66BB6A", width=1, dash="dash"))
        fig.add_shape(type="line", x0=df.index[-120], y0=s_forte, x1=df.index[-1], y1=s_forte, line=dict(color="#1B5E20", width=2, dash="solid"))
    
    fig.update_layout(title=f"{ticker} - {tipo_grafico}", template="plotly_dark", height=500, xaxis_rangeslider_visible=False, margin=dict(l=50, r=50, t=50, b=50), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# === NOVO: FUNÇÃO PARA GERAR BACKTEST ===
def gerar_grafico_backtest(df_d, df_w, ticker):
    """
    Recalcula o Score Vector 3 para os últimos 20 dias e plota um gráfico combo.
    Nota: Para performance, neste backtest usamos uma aproximação do semanal e ignoramos o intraday.
    """
    df_calc = df_d.tail(30).copy() # Pega 30 para garantir 20 de exibição
    
    scores = []
    dates = []
    prices = []
    
    for i in range(len(df_calc)):
        if i < 1: continue # Pula o primeiro
        
        # Simula o 'hoje' histórico
        row = df_calc.iloc[i]
        date_curr = df_calc.index[i]
        
        # Pontuação Simplificada para Histórico (Daily + Price Action + Weekly Approx)
        # Score Maximo aqui será 5 (ignorando Intraday 120m que é pesado para recalcular)
        score_dia = 0
        
        # 1. Tendência Diária
        if (row['Close'] > row['EMA21']) and (row['EMA21'] > row['SMA50']): score_dia += 2 # Alta
        elif (row['Close'] < row['EMA21']) and (row['EMA21'] < row['SMA50']): score_dia += 2 # Baixa
        
        # 2. MACD
        if (score_dia > 0): # Só pontua MACD se tiver tendência
            # Se tendência alta e MACD > Signal
            if (row['Close'] > row['EMA21']) and (row['MACD'] > row['MACD_Signal']): score_dia += 1
            # Se tendência baixa e MACD < Signal
            elif (row['Close'] < row['EMA21']) and (row['MACD'] < row['MACD_Signal']): score_dia += 1
            
        # 3. Price Action (Rompimento de 20 dias atrás)
        if (score_dia > 0):
            # Se alta e rompeu máxima de 20 dias
            if (row['Close'] > row['EMA21']) and (row['Close'] > row['Resistencia_Hist']): score_dia += 1
            # Se baixa e rompeu mínima de 20 dias
            elif (row['Close'] < row['EMA21']) and (row['Close'] < row['Suporte_Hist']): score_dia += 1
            
        # 4. Semanal (Aproximação: Pega o semanal da época)
        # Convertendo para timezone naive para comparação segura
        try:
            date_lookup = date_curr.replace(tzinfo=None)
            # Encontra a vela semanal que contem este dia
            # Como df_w tem index date, usamos asof ou reindex. Simplificando:
            w_idx = df_w.index.get_indexer([date_lookup], method='pad')[0]
            if w_idx != -1:
                w_row = df_w.iloc[w_idx]
                if (score_dia >= 2): # Só verifica semanal se diário tiver tendência
                    if (row['Close'] > row['EMA21']) and (w_row['Close'] > w_row['EMA21']): score_dia += 1
                    elif (row['Close'] < row['EMA21']) and (w_row['Close'] < w_row['EMA21']): score_dia += 1
        except:
            pass # Se falhar a data, mantém o score
            
        scores.append(score_dia)
        dates.append(date_curr)
        prices.append(row['Close'])
        
    # Criação do Gráfico Combo
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar Chart (Scores)
    colors = ['#EF5350' if s < 3 else ('#FFEE58' if s == 3 else '#66BB6A') for s in scores]
    
    fig.add_trace(
        go.Bar(x=dates, y=scores, name="Score Histórico", marker_color=colors, opacity=0.6),
        secondary_y=False,
    )

    # Line Chart (Price)
    fig.add_trace(
        go.Scatter(x=dates, y=prices, name="Preço", line=dict(color='white', width=2)),
        secondary_y=True,
    )

    fig.update_layout(
        title=f"Backtest Visual: Score vs Preço ({ticker})",
        template="plotly_dark",
        height=400,
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Score Vector 3", range=[0, 6], secondary_y=False)
    fig.update_yaxes(title_text="Preço (R$)", secondary_y=True)

    return fig

# ================= INTERFACE =================
st.title("💠 VECTOR 3")
st.markdown("### *Algorithmic Market Scanner*")

# --- SIDEBAR ---
st.sidebar.markdown("## 💠 Vector 3")
st.sidebar.markdown("---")
st.sidebar.caption("Configuração de Análise")

selecao = IBXX_FULL_LIST
check_all = st.sidebar.checkbox("Analisar IBrX 100", value=False)
if not check_all:
    selecao = st.sidebar.multiselect("Carteira:", IBXX_FULL_LIST, default=["PETR4", "VALE3", "BOVA11", "MGLU3", "PRIO3", "BBAS3"])

st.sidebar.markdown("---")
botao_analise = st.sidebar.button("🔍 SCANEAR MERCADO", type="primary")

st.sidebar.markdown("### Metodologia")
st.sidebar.markdown("""
**1. Triple Screen:**
* **Macro (W):** Filtro de Tendência.
* **Gatilho (D):** Médias + MACD.
* **Timing (120m):** Sintonia fina.

**2. Price Action:**
* Suportes e Resistências calculados dinamicamente via Pivôs.
""")

# --- BOAS VINDAS ---
if not st.session_state.analise_realizada and not botao_analise:
    st.markdown("---")
    st.markdown("""
    ### 🎯 Pare de Adivinhar. Comece a Calcular.
    O **Vector 3** elimina o ruído do mercado e foca na estrutura de preço. 
    Nossa engine processa múltiplos tempos gráficos para encontrar a confluência perfeita.
    """)
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🌊 A Maré (Trend)")
        st.info("**Filtro Macro:** Só operamos a favor da tendência principal. Se o Semanal diz 'não', o Diário obedece.")
    with col2:
        st.subheader("🌊 A Onda (Setup)")
        st.warning("**Gatilho Técnico:** Cruzamento de médias, Momentum (MACD) e quebra de estrutura (Price Action).")
    with col3:
        st.subheader("🎯 O Timing (Entry)")
        st.success("**Sintonia Fina:** O gráfico de 120min confirma se o momento exato da entrada é agora.")
    st.divider()
    st.caption("👈 Selecione seus ativos na barra lateral e clique em 'SCANEAR MERCADO' para iniciar.")

# --- PROCESSAMENTO ---
if botao_analise:
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
    st.rerun()

# --- RESULTADOS ---
if st.session_state.analise_realizada:
    st.title("💠 VECTOR 3 | Resultados")
    df_res = pd.DataFrame(st.session_state.dados_analise)
    cols = ['Ativo', 'Preço', 'Score', 'Setup', 'Stop_Tecnico', 'Observacoes']
    
    if not df_res.empty:
        mask_alta = (df_res['Direção'] == "ALTA") & (df_res['Score'] >= 3)
        mask_baixa = (df_res['Direção'] == "BAIXA") & (df_res['Score'] >= 3)
        mask_aguardando = ~mask_alta & ~mask_baixa
        
        df_alta = df_res[mask_alta].sort_values('Score', ascending=False)
        df_baixa = df_res[mask_baixa].sort_values('Score', ascending=False)
        df_aguardando = df_res[mask_aguardando].sort_values('Score', ascending=False)
        
        c1, c2 = st.columns(2)
        with c1: 
            st.success(f"🚀 ALTA (Score 3+): {len(df_alta)}")
            if not df_alta.empty: st.dataframe(df_alta[cols], use_container_width=True, hide_index=True)
        with c2: 
            st.error(f"🩸 BAIXA (Score 3+): {len(df_baixa)}")
            if not df_baixa.empty: st.dataframe(df_baixa[cols], use_container_width=True, hide_index=True)
        
        with st.expander(f"⏳ Radar de Observação / Aguardando ({len(df_aguardando)})", expanded=False):
            st.write("Ativos com tendência indefinida ou Score insuficiente (< 3).")
            if not df_aguardando.empty:
                st.dataframe(df_aguardando[['Ativo', 'Preço', 'Direção', 'Score', 'Observacoes']], use_container_width=True, hide_index=True)
        
        st.divider()
        st.subheader("🕵️‍♂️ Detalhamento & Gráficos")
        escolha = st.selectbox("Selecione Ativo para Raio-X:", df_res['Ativo'].tolist())
        
        if escolha:
            d_ativo = next(i for i in st.session_state.dados_analise if i["Ativo"] == escolha)
            w, d, h = d_ativo['analise_w'], d_ativo['analise_d'], d_ativo['analise_120']
            
            # ABAS COM BACKTEST INCLUÍDO
            tab_relatorio, tab_score, tab_data, tab_chart, tab_backtest = st.tabs([
                "📋 Relatório IA", "📝 Scorecard", "🔢 Dados Estruturais", "📊 Gráfico", "🔙 Backtest (20d)"
            ])
            
            with tab_relatorio:
                relatorio = gerar_relatorio_textual(d_ativo)
                st.markdown(relatorio)
                st.info(f"**Plano Sugerido:** Entrada próxima a R\$ {d_ativo['Preço']:.2f}, buscando Alvo em R\$ {d_ativo['Alvo']:.2f} com proteção em R\$ {d_ativo['Stop_Tecnico']:.2f}.")

            with tab_score:
                st.markdown("#### 🎯 Métricas de Risco")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Preço", f"R$ {d_ativo['Preço']:.2f}")
                m2.metric("Stop (1.5x ATR)", f"R$ {d_ativo['Stop_Tecnico']:.2f}")
                m3.metric("Alvo (3x ATR)", f"R$ {d_ativo['Alvo']:.2f}")
                m4.metric("Payoff (Risco/Retorno)", f"1 : {d_ativo['Payoff']:.1f}", delta="Aprovado" if d_ativo['Payoff'] >= 2.0 else "Atenção")
                st.divider()
                st.markdown("#### 📝 Composição do Score")
                bk = d_ativo['Breakdown']
                s1, s2, s3, s4, s5 = st.columns(5)
                def fs(v): return f"✅ +{v}" if v>0 else "❌ 0"
                s1.metric("Tendência (D)", fs(bk['Tendência Diária'])); s2.metric("MACD (D)", fs(bk['MACD Diário'])); s3.metric("Semanal (W)", fs(bk['Confluência Semanal']))
                s4.metric("Intraday", fs(bk['Confluência Intraday'])); s5.metric("Price Action", fs(bk['Price Action']))

            with tab_data:
                col_w, col_d, col_h = st.columns(3)
                with col_w:
                    st.info("📅 SEMANAL"); st.write(f"Viés: **{w['Viés']}**"); st.caption(f"Motivo: {w['Motivos']}")
                with col_d:
                    st.warning("📆 DIÁRIO"); st.write(f"Viés: **{d['Viés']}**")
                    st.markdown("**Resistências:**"); st.code(f"Imed: {d['Res_Imediata']:.2f}\nForte:{d['Res_Forte']:.2f}")
                    st.markdown("**Suportes:**"); st.code(f"Imed: {d['Sup_Imediato']:.2f}\nForte:{d['Sup_Forte']:.2f}")
                with col_h:
                    st.success("⏱️ 120 MIN"); st.write(f"Viés: **{h['Viés']}**")

            with tab_chart:
                col_sel, _ = st.columns([1, 3])
                with col_sel:
                    tf_selecionado = st.radio("Tempo Gráfico:", ["Diário", "Semanal", "120 Minutos"], horizontal=True)
                if tf_selecionado == "Semanal": fig = criar_grafico_dinamico(d_ativo['df_chart_w'], escolha, d, "Semanal")
                elif tf_selecionado == "Diário": fig = criar_grafico_dinamico(d_ativo['df_chart_d'], escolha, d, "Diário")
                else: fig = criar_grafico_dinamico(d_ativo['df_chart_120'], escolha, d, "120 Minutos")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab_backtest:
                st.markdown("#### ⏳ Histórico de Pontuação (Últimos 20 Pregões)")
                st.caption("Visualiza a correlação entre o Score Vector 3 (Barras) e o Preço (Linha Branca). Barras Verdes indicam Score >= 4.")
                fig_bt = gerar_grafico_backtest(d_ativo['df_chart_d'], d_ativo['df_chart_w'], escolha)
                st.plotly_chart(fig_bt, use_container_width=True)
