from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2 import OperationalError
from behaviors import BehaviorEvent
from events import EventRecord

class DatabaseManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection = None
    
    def connect(self):
        try:
            self.connection = psycopg2.connect(
                host=self.config.get("host", "localhost"),
                port=self.config.get("port", 5432),
                dbname=self.config.get("dbname", "video_analysis"),
                user=self.config.get("user", "admin"),
                password=self.config.get("password", "password")
            )
            return True
        except OperationalError as e:
            print(f"Database connection failed: {e}")
            return False
    
    def create_tables(self):
        if not self.connection:
            if not self.connect():
                return False
        
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id SERIAL PRIMARY KEY,
                    event_type VARCHAR(50) NOT NULL,
                    track_id INTEGER,
                    class_name VARCHAR(50),
                    bbox JSONB,
                    timestamp FLOAT NOT NULL,
                    description TEXT,
                    severity VARCHAR(20),
                    zone_name VARCHAR(100),
                    video_path VARCHAR(500),
                    frame_indices JSONB,
                    vlm_description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id SERIAL PRIMARY KEY,
                    event_id INTEGER REFERENCES events(id),
                    bbox JSONB,
                    confidence FLOAT,
                    class_id INTEGER,
                    class_name VARCHAR(50),
                    timestamp FLOAT
                )
            """)
            
            self.connection.commit()
            cursor.close()
            return True
        except OperationalError as e:
            print(f"Failed to create tables: {e}")
            return False
    
    def insert_event(self, record: EventRecord) -> int:
        if not self.connection:
            if not self.connect():
                return -1
        
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT INTO events (
                    event_type, track_id, class_name, bbox, timestamp,
                    description, severity, zone_name, video_path, 
                    frame_indices, vlm_description
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                record.event.event_type,
                record.event.track_id,
                record.event.class_name,
                record.event.bbox,
                record.event.timestamp,
                record.event.description,
                record.event.severity,
                record.event.zone_name,
                record.video_path,
                record.frame_indices,
                record.vlm_description
            ))
            
            event_id = cursor.fetchone()[0]
            self.connection.commit()
            cursor.close()
            return event_id
        except OperationalError as e:
            print(f"Failed to insert event: {e}")
            return -1
    
    def get_events(self, event_type: str = None, severity: str = None) -> List[Dict[str, Any]]:
        if not self.connection:
            if not self.connect():
                return []
        
        try:
            cursor = self.connection.cursor()
            
            query = "SELECT * FROM events WHERE 1=1"
            params = []
            
            if event_type:
                query += " AND event_type = %s"
                params.append(event_type)
            
            if severity:
                query += " AND severity = %s"
                params.append(severity)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            columns = [desc[0] for desc in cursor.description]
            result = [dict(zip(columns, row)) for row in rows]
            
            cursor.close()
            return result
        except OperationalError as e:
            print(f"Failed to get events: {e}")
            return []
    
    def close(self):
        if self.connection:
            self.connection.close()