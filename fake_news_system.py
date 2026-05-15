import sys
import os
import json
import time
import random
from datetime import datetime
import joblib
import pandas as pd
import altair as alt
from pade.acl.aid import AID
from pade.acl.messages import ACLMessage
from pade.behaviours.protocols import Behaviour, TimedBehaviour
from pade.core.agent import Agent
from pade.misc.utility import display_message, start_loop
from csv_reader import NewsCSVReader

# CONSTANTES
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(_SCRIPT_DIR, "data/Fake.csv")
READ_INTERVAL = 3.0
DATA_FILE = "fake_news_data.json"

# BEHAVIOURS DEL AGENTE PERCEPCION
class ReadNewsFromCSV(TimedBehaviour):
    def __init__(self, agent: "PerceptionAgent", interval: float):
        super().__init__(agent, interval)

    def on_time(self):
        if not self.agent.csv_reader.has_more():
            display_message(self.agent.aid.name, "CSV terminado. Reiniciando puntero.")
            self.agent.csv_reader.reset()
            return

        news = self.agent.csv_reader.get_next()
        display_message(
            self.agent.aid.name,
            f"Noticia leida ID={news.get('id', 'N/A')} | {news.get('title', '')[:50]}"
        )

        msg = self._build_inform(NewsCSVReader.to_json(news))
        self.agent.send(msg)
        display_message(
            self.agent.aid.name,
            f"Mensaje enviado a {self.agent.classifier_name} (conv_id={msg.conversation_id})"
        ) 

        super().on_time()

    def _build_inform(self, payload: str) -> ACLMessage:
        msg = ACLMessage(ACLMessage.INFORM)
        msg.set_protocol(ACLMessage.FIPA_REQUEST_PROTOCOL)
        msg.set_ontology("fake-news-detection")
        msg.set_language("JSON")
        msg.set_content(payload)
        msg.add_receiver(AID(name=self.agent.classifier_name))
        return msg

class DFRegistrationBehaviour(Behaviour):
    def action(self):
        service = {
            "service-type": "data-provider",
            "service-name": "perception-service",
            "agent-name": self.agent.aid.name,
            "ontology": "fake-news-detection",
        }
        msg = ACLMessage(ACLMessage.SUBSCRIBE)
        msg.set_ontology("fipa-df")
        msg.set_language("JSON")
        msg.set_content(NewsCSVReader.to_json(service))
        msg.add_receiver(AID(name="DF@localhost:8000"))
        self.agent.send(msg)
        display_message(self.agent.aid.name, "Registro enviado al DF: perception-service")

    def done(self):
        return True

# AGENTE DE PERCEPCION
class PerceptionAgent(Agent):
    def __init__(self, aid: AID, csv_path: str, classifier_name: str):
        super().__init__(aid)
        self.csv_path = csv_path
        self.classifier_name = classifier_name
        self.csv_reader: NewsCSVReader = NewsCSVReader(csv_path)

    def on_start(self):
        super().on_start()
        display_message(self.aid.name, "=== Agente de Percepcion iniciando ===")

        try:
            count = self.csv_reader.load()
            display_message(self.aid.name, f"CSV cargado: {count} noticias disponibles.")
        except (FileNotFoundError, ValueError) as e:
            display_message(self.aid.name, f"ERROR al cargar CSV: {e}")
            return

        self.behaviours.append(DFRegistrationBehaviour(self))
        self.behaviours.append(ReadNewsFromCSV(self, READ_INTERVAL))

        display_message(
            self.aid.name,
            f"Agente listo. Enviando noticias cada {READ_INTERVAL}s -> {self.classifier_name}"
        )

    def react(self, message):
        super().react(message)

        if getattr(message, 'system_message', False):
            return

        if message.performative == ACLMessage.CONFIRM:
            try:
                content = json.loads(message.content)
                # Modificado ligeramente para evitar fallos si el clasificador manda el resultado completo en lugar de solo status
                display_message(
                    self.aid.name,
                    f"Confirmacion recibida: ID={content.get('id')} process_status=OK"
                )
            except Exception:
                display_message(
                    self.aid.name,
                    f"Confirmacion recibida: {str(message.content)[:80]}"
                )

# AGENTE CLASIFICADOR
class ClassifierAgent(Agent):
    def __init__(self, aid: AID, gui_agent_aid: AID):
        super().__init__(aid=aid, debug=False)
        self.model = joblib.load("./model/fake_news_model.joblib")
        self.gui_agent_aid = gui_agent_aid

    def classify_news(self, news):
        title = str(news.get("title", ""))
        text = str(news.get("text", ""))

        model_input = title + " " + text

        pred = self.model.predict([model_input])[0]

        if hasattr(self.model, "predict_proba"):
            confidence = float(max(self.model.predict_proba([model_input])[0]))
        else:
            confidence = 0.5

        result = {
            "id": news.get("id"),
            "title": title,
            "source": news.get("source", "unknown"),
            "is_fake": bool(pred),
            "confidence": round(confidence, 2),
            "timestamp": datetime.now().isoformat()
        }

        return result

    def react(self, message):
        super().react(message)
        if message.ontology != "fake-news-detection":
            return

        if message.performative != ACLMessage.INFORM:
            return

        try:
            news = json.loads(message.content)
            result = self.classify_news(news)

            # Integration point: Send CONFIRM back to PerceptionAgent
            reply = ACLMessage(ACLMessage.CONFIRM)
            reply.set_ontology("fake-news-detection")
            reply.set_language("JSON")
            reply.set_conversation_id(message.conversation_id)
            reply.set_content(json.dumps(result))

            if message.sender is not None:
                reply.add_receiver(message.sender)
            self.send(reply)

            # Integration point: Send INFORM to GUI_Agent
            gui_msg = ACLMessage(ACLMessage.INFORM)
            gui_msg.set_ontology("fake_news_ontology")
            gui_msg.set_language("JSON")
            gui_msg.set_content(json.dumps(result))
            
            if self.gui_agent_aid is not None:
                gui_msg.add_receiver(self.gui_agent_aid)
            self.send(gui_msg)

            display_message(
                self.aid.name,
                f"[CLASIFICADO] ID={news.get('id')} fake={result['is_fake']} "
                f"confianza={result['confidence']}"
            )

        except Exception as e:
            display_message(self.aid.name, f"[ERROR] {e}")


# AGENTE GUI
class GUI_Agent(Agent):
    def __init__(self, aid: AID):
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
        if message.ontology != 'fake_news_ontology':
            return
            
        display_message(self.aid.localname, f'Recibidos datos de {message.sender.name}')
        
        try:
            new_data = json.loads(message.content)
            self.data_list.append(new_data)
            self._save_data()
        except Exception as e:
            display_message(self.aid.localname, f"Error al procesar el mensaje JSON: {e}")

# RENDERIZADO DEL GUI
def render_gui():
    if st is None:
        print("Streamlit no está instalado. Instálalo con 'pip install streamlit'")
        return

    st.set_page_config(page_title="Fake News Dashboard", layout="wide", page_icon="📰")
    st.title("📰 Análisis de Noticias Falsas (Multiagente)")
    st.markdown("Dashboard alimentado por un Agente PADE que recopila datos de otros agentes del sistema.")

    with st.sidebar:
        st.header("Configuración del Dashboard")
        auto_refresh = st.checkbox("Recarga automática", value=True)
        refresh_interval = st.slider("Intervalo de recarga (s)", 1, 10, 3) if auto_refresh else 0
        
        if st.button("Recargar ahora", width='stretch'):
            st.rerun()
            
        st.divider()
        st.markdown("**Instrucciones de Uso:**\n\n1. Inicia el sistema PADE en una terminal:\n```bash\npade start-runtime fake_news_system.py\n```\n2. En **otra terminal**, lanza Streamlit:\n```bash\nstreamlit run fake_news_system.py\n```")
        
        if st.button("Limpiar Datos"):
            if os.path.exists(DATA_FILE):
                os.remove(DATA_FILE)
            st.rerun()

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

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

    total_news = len(df)
    fake_news = len(df[df['is_fake'] == True])
    real_news = total_news - fake_news
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Analizadas", total_news)
    col2.metric("Noticias Falsas (Fake)", fake_news, delta=f"{(fake_news/total_news)*100:.1f}%" if total_news > 0 else "0%", delta_color="inverse")
    col3.metric("Noticias Reales", real_news, delta=f"{(real_news/total_news)*100:.1f}%" if total_news > 0 else "0%", delta_color="normal")

    st.markdown("---")

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


if __name__ == '__main__':
    # Punto de entrada dual: Streamlit o PADE
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        st = None

    is_streamlit = False
    if st is not None:
        try:
            if get_script_run_ctx() is not None:
                is_streamlit = True
        except Exception:
            pass

    if is_streamlit:
        render_gui()
    else:
        base_port = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

        perception_port  = base_port
        classifier_port  = base_port + 1
        gui_port         = base_port + 2

        perception_aid = AID(name=f"perception_agent@localhost:{perception_port}")
        classifier_aid = AID(name=f"classifier_agent@localhost:{classifier_port}")
        gui_aid = AID(name=f"gui_agent@localhost:{gui_port}")

        perception = PerceptionAgent(
            aid=perception_aid,
            csv_path=CSV_FILE_PATH,
            classifier_name=classifier_aid.name,
        )
        
        classifier = ClassifierAgent(
            aid=classifier_aid,
            gui_agent_aid=gui_aid
        )
        
        gui_agent = GUI_Agent(
            aid=gui_aid
        )

        start_loop([perception, classifier, gui_agent])
