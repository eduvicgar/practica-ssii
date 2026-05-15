import sys
import os
import json
from pade.acl.aid import AID
from pade.acl.messages import ACLMessage
from pade.behaviours.protocols import Behaviour, TimedBehaviour
from pade.core.agent import Agent
from pade.misc.utility import display_message, start_loop
from csv_reader import NewsCSVReader

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(_SCRIPT_DIR, "Fake.csv")
READ_INTERVAL = 10.0
AGENT_NAME = "perception_agent"
CLASSIFIER_AGENT_NAME = "classifier_agent@localhost:{port}"

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
            f"Noticia leida ID={news['id']} | {news['title'][:50]}"
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
        return True  # se ejecuta una sola vez

class PerceptionAgent(Agent):
    def __init__(self, aid: AID, csv_path: str = CSV_FILE_PATH, classifier_name):
        super().__init__(aid)
        self.csv_path = csv_path
        self.classifier_name = classifier_name

        # Instancia del lector de CSV
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

        # Ignorar mensajes de sistema
        if getattr(message, 'system_message', False):
            return

        # Procesar confirmaciones del clasificador
        if message.performative == ACLMessage.CONFIRM:
            try:
                content = json.loads(message.content)
                display_message(
                    self.aid.name,
                    f"Confirmacion recibida: ID={content.get('id')} status={content.get('status')}"
                )
            except Exception:
                display_message(
                    self.aid.name,
                    f"Confirmacion recibida: {str(message.content)[:80]}"
                )


# =============================================================================
# CLASIFICADOR DUMMY REEMPLAZAR
# =============================================================================

class ClassifierDummy(Agent):
    def on_start(self):
        super().on_start()
        display_message(self.aid.name, "=== Clasificador DUMMY listo. Esperando noticias ===")

    def react(self, message):
        super().react(message)

        # Ignorar mensajes de sistema
        if getattr(message, "system_message", False):
            return

        # Solo procesar INFORM con ontologia fake-news-detection
        if message.performative != ACLMessage.INFORM:
            return

        if getattr(message, "ontology", None) != "fake-news-detection":
            return

        # Intentar parsear la noticia
        news = {}
        try:
            raw = message.content
            if raw:
                news = json.loads(raw)
            display_message(
                self.aid.name,
                f"[RECIBIDO] ID={news.get('id')} | {news.get('title', '')[:60]}"
            )
        except Exception as e:
            display_message(self.aid.name, f"[RECIBIDO] contenido no JSON: {e}")

        # Responder CONFIRM con el mismo conversation_id
        reply = ACLMessage(ACLMessage.CONFIRM)
        reply.set_ontology("fake-news-detection")
        reply.set_language("JSON")
        reply.set_conversation_id(message.conversation_id)
        reply.set_content(json.dumps({"status": "ok", "id": news.get("id")}))
        reply.add_receiver(message.sender)
        self.send(reply)
        display_message(
            self.aid.name,
            f"[CONFIRM enviado] conv_id={message.conversation_id}"
        )

if __name__ == "__main__":
    # Cuando se ejecuta con: pade start-runtime --port 2000 perception_agent.py
    # PADE pasa el puerto base como sys.argv[1]
    if len(sys.argv) >= 2:
        try:
            base_port = int(sys.argv[1])
        except ValueError:
            print(f"Error: se esperaba un puerto numerico, se recibio '{sys.argv[1]}'")
            sys.exit(1)
    else:
        base_port = 2000

    perception_port = base_port
    classifier_port = base_port + 1

    classifier_full_name = f"classifier_agent@localhost:{classifier_port}"

    perception = PerceptionAgent(
        aid=AID(name=f"{AGENT_NAME}@localhost:{perception_port}"),
        csv_path=CSV_FILE_PATH,
        classifier_name=classifier_full_name,
    )
    classifier = ClassifierDummy(
        AID(name=f"classifier_agent@localhost:{classifier_port}")
    )

    start_loop([perception, classifier])