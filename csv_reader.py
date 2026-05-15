import csv
import json
from datetime import datetime
from pathlib import Path

class NewsCSVReader:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self._news = []
        self._index = 0
        
    def load(self):
        if not self.filepath.exists():
            print(self.filepath)
            raise FileNotFoundError(f"CSV no encontrado: {self.filepath}")
                
        with open(self.filepath, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            
            self._news = [
                self._build(i, row)
                for i, row in enumerate(reader)
                if row.get("title") or row.get("text")
            ]
        
        self._index = 0
        return len(self._news)    

    def _build(self, i, row):
        return {
            "id": i,
            "title": (row.get("title") or "").strip(),
            "text": (row.get("text") or "").strip(),
            "subject": (row.get("subject") or "").strip(),
            "date": (row.get("date") or "").strip(),
            "ingested_at": datetime.now().isoformat()
        }
        
    def has_more(self):
        return self._index < len(self._news)
    
    def get_next(self):
        if not self.has_more():
            return None
        news = self._news[self._index]
        self._index += 1
        return news
    
    def reset(self):
        self._index = 0
    
    @staticmethod    
    def to_json(news):
        return json.dumps(news, indent=2)
    
if __name__ == "__main__":
    reader = NewsCSVReader("Fake.csv")
    reader.load()
    
    for i in range(5):
        print(NewsCSVReader.to_json(reader.get_next()))
        
    
    
        