import sys
import json
import os
import random
import time
from datetime import datetime
import pandas as pd
import altair as alt

try:
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except ImportError:
    st = None

# PADE imports
from pade.core.agent import Agent
from pade.acl.messages import ACLMessage
from pade.acl.aid import AID
from pade.behaviours.protocols import TimedBehaviour
from pade.misc.utility import display_message, start_loop

DATA_FILE = "fake_news_data.json"

# ==========================================
# 1. AGENTE PADE PARA RECOLECCIÓN DE DATOS
# ==========================================
class GUI_Agent(Agent):
    """
    Agente encargado de recibir los veredictos de fake news de otros agentes
    y guardarlos para que Streamlit los muestre.
    """
    def __init__(self, aid):
        super().__init__(aid=aid, debug=False)
        self.data_list = []
        self._load_data()

    def _load_data(self):
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, "r", encoding='utf-8') as f:
                    self.data_list = json.load(f)
            except Exception:
                self.data_list = []
        else:
            self.data_list = []

    def _save_data(self):
        with open(DATA_FILE, "w", encoding='utf-8') as f:
            json.dump(self.data_list, f, indent=4)

    def on_start(self):
        super().on_start()
        display_message(self.aid.localname, 'GUI Agent iniciado. Esperando datos de noticias...')

    def react(self, message):
        super().react(message)
        # Solo reaccionar a la ontología de nuestro sistema
        if message.ontology != 'fake_news_ontology':
            return
            
        display_message(self.aid.localname, f'Recibidos datos de {message.sender.name}')
        
        try:
            new_data = json.loads(message.content)
            self.data_list.append(new_data)
            self._save_data()
        except Exception as e:
            display_message(self.aid.localname, f"Error al procesar el mensaje JSON: {e}")

# ==========================================
# 2. AGENTE HIPOTÉTICO (SIMULADOR)
# ==========================================
class SendDataBehaviour(TimedBehaviour):
    """
    Comportamiento temporal que envía resultados falsos cada N segundos.
    """
    def __init__(self, agent, time, receiver_aid):
        super(SendDataBehaviour, self).__init__(agent, time)
        self.receiver_aid = receiver_aid
        self.sources = ["Twitter", "Facebook", "El Mundo", "WhatsApp", "New York Times"]
        self.titles = [
            "Descubren agua en Marte",
            "La inteligencia artificial reemplazará a los programadores",
            "Nueva subida de impuestos anunciada",
            "El café prolonga la vida, según nuevo estudio",
            "Imágenes exclusivas del nuevo iPhone 20"
        ]

    def on_time(self):
        super(SendDataBehaviour, self).on_time()
        
        is_fake = random.choice([True, False])
        confidence = random.uniform(0.7, 0.99)
        title = random.choice(self.titles)
        source = random.choice(self.sources)
        timestamp = datetime.now().isoformat()
        
        data = {
            "title": title,
            "source": source,
            "is_fake": is_fake,
            "confidence": round(confidence, 2),
            "timestamp": timestamp
        }
        
        message = ACLMessage(ACLMessage.INFORM)
        message.add_receiver(self.receiver_aid)
        message.set_ontology('fake_news_ontology')
        message.set_content(json.dumps(data))
        self.agent.send(message)
        
        display_message(self.agent.aid.localname, f"Datos de prueba enviados: {'Fake' if is_fake else 'Real'}")

class MockSenderAgent(Agent):
    """
    Agente hipotético que simula el resto de tu sistema.
    """
    def __init__(self, aid, receiver_aid):
        super().__init__(aid=aid, debug=False)
        self.receiver_aid = receiver_aid

    def on_start(self):
        super().on_start()
        display_message(self.aid.localname, 'Agente Hipotético iniciado. Enviando datos cada 5 segundos...')
        # Enviar datos simulados cada 5 segundos
        self.behaviours.append(SendDataBehaviour(self, 5.0, self.receiver_aid))


# ==========================================
# 3. INTERFAZ GRÁFICA CON STREAMLIT
# ==========================================
def render_gui():
    if st is None:
        print("Streamlit no está instalado. Instálalo con 'pip install streamlit'")
        return

    st.set_page_config(page_title="Fake News Dashboard", layout="wide", page_icon="📰")
    st.title("📰 Análisis de Noticias Falsas (Multiagente)")
    st.markdown("Dashboard alimentado por un Agente PADE que recopila datos de otros agentes del sistema.")

    # Sidebar para configuración
    with st.sidebar:
        st.header("Configuración del Dashboard")
        auto_refresh = st.checkbox("Recarga automática", value=True)
        refresh_interval = st.slider("Intervalo de recarga (s)", 1, 10, 3) if auto_refresh else 0
        
        if st.button("Recargar ahora", width='stretch'):
            st.rerun()
            
        st.divider()
        st.markdown("**Instrucciones de Uso:**\n\n1. Inicia el sistema PADE en una terminal:\n```bash\npade start-runtime agent_gui.py\n```\n2. En **otra terminal**, lanza Streamlit:\n```bash\nstreamlit run agent_gui.py\n```")
        
        if st.button("Limpiar Datos"):
            if os.path.exists(DATA_FILE):
                os.remove(DATA_FILE)
            st.rerun()

    # Carga de datos
    if not os.path.exists(DATA_FILE):
        st.info("⏳ Esperando datos... Inicia el entorno PADE para comenzar la recolección de noticias.")
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
        return

    try:
        with open(DATA_FILE, "r", encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        data = []

    if not data:
        st.info("El archivo de datos está vacío. Esperando mensajes de los agentes...")
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
        return

    # Procesar dataframe
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

    # Métricas principales
    total_news = len(df)
    fake_news = len(df[df['is_fake'] == True])
    real_news = total_news - fake_news
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Analizadas", total_news)
    col2.metric("Noticias Falsas (Fake)", fake_news, delta=f"{(fake_news/total_news)*100:.1f}%" if total_news > 0 else "0%", delta_color="inverse")
    col3.metric("Noticias Reales", real_news, delta=f"{(real_news/total_news)*100:.1f}%" if total_news > 0 else "0%", delta_color="normal")

    st.markdown("---")

    # Gráficos
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Distribución de Fiabilidad")
        counts = df['is_fake'].value_counts().reset_index()
        counts.columns = ['Es Falsa', 'Cantidad']
        counts['Clasificación'] = counts['Es Falsa'].map({True: 'Fake News', False: 'Verdadera'})
        
        chart = alt.Chart(counts).mark_arc(innerRadius=60).encode(
            theta=alt.Theta(field="Cantidad", type="quantitative"),
            color=alt.Color(field="Clasificación", type="nominal", scale=alt.Scale(domain=['Verdadera', 'Fake News'], range=['#2ecc71', '#e74c3c'])),
            tooltip=['Clasificación', 'Cantidad']
        ).properties(height=350)
        st.altair_chart(chart, width='stretch')

    with colB:
        st.subheader("Distribución de Confianza del Modelo")
        chart2 = alt.Chart(df).mark_bar(opacity=0.8).encode(
            x=alt.X('confidence:Q', bin=alt.Bin(maxbins=10), title='Nivel de Confianza'),
            y=alt.Y('count():Q', title='Cantidad de Artículos'),
            color=alt.Color('is_fake:N', scale=alt.Scale(domain=[False, True], range=['#2ecc71', '#e74c3c']), title='Es Fake')
        ).properties(height=350)
        st.altair_chart(chart2, width='stretch')

    st.subheader("Últimas Noticias Procesadas")
    st.dataframe(
        df[['timestamp', 'title', 'source', 'confidence', 'is_fake']].sort_values(by='timestamp', ascending=False),
        width='stretch',
        hide_index=True,
        column_config={
            "timestamp": "Fecha y Hora",
            "title": "Título de la Noticia",
            "source": "Fuente",
            "confidence": st.column_config.ProgressColumn("Confianza", min_value=0, max_value=1, format="%.2f"),
            "is_fake": "Es Falsa"
        }
    )

    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


# ==========================================
# PUNTO DE ENTRADA (Maneja Streamlit o PADE)
# ==========================================
if __name__ == '__main__':
    is_streamlit = False
    
    # Comprobar de forma segura si estamos dentro de Streamlit
    if st is not None:
        try:
            if get_script_run_ctx() is not None:
                is_streamlit = True
        except Exception:
            pass

    if is_streamlit:
        # Modo Interfaz Gráfica (ejecutado por `streamlit run`)
        render_gui()
    else:
        # Modo Sistema Multiagente (ejecutado por `pade start-runtime`)
        ams_config = {'name': 'localhost', 'port': 8000}
        
        # PADE inyecta el puerto base en sys.argv[1]
        base_port = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
        
        gui_port = base_port
        mock_port = base_port + 1000
        
        gui_aid = AID(name=f'gui_agent_{gui_port}@localhost:{gui_port}')
        mock_aid = AID(name=f'mock_agent_{mock_port}@localhost:{mock_port}')
        
        gui_agent = GUI_Agent(gui_aid)
        mock_agent = MockSenderAgent(mock_aid, gui_aid)
        
        for agent in (gui_agent, mock_agent):
            agent.update_ams(ams_config)
            
        start_loop([gui_agent, mock_agent])
