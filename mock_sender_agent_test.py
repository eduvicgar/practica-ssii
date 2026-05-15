import pandas as pd
import joblib
import json
from datetime import datetime

from pade.acl.messages import ACLMessage
from pade.acl.aid import AID

from mock_sender_agent import MockSenderAgent


model = joblib.load("./model/fake_news_model.joblib")


df = pd.read_csv("./data/fake_or_real_news.csv")


agent = MockSenderAgent(AID(name="mock@test"))


for i in range(5):

    row = df.iloc[i]

    payload = json.dumps({
        "title": str(row["title"]),
        "text": str(row["text"]),
        "source": str(row.get("source", "unknown"))
    })

   
    msg = ACLMessage(ACLMessage.INFORM)
    msg.set_ontology("fake_news_ontology")
    msg.set_content(payload)
    msg.sender = AID(name="tester@localhost:50000")


    agent.react(msg)