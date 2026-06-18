import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any
import hashlib

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

DB_PATH = "./data/smart_video.db"

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            url TEXT,
            enabled INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            message TEXT NOT NULL,
            camera_id INTEGER,
            frame_path TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (camera_id) REFERENCES cameras(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER,
            filename TEXT NOT NULL,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            duration INTEGER,
            FOREIGN KEY (camera_id) REFERENCES cameras(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            person_count INTEGER DEFAULT 0,
            vehicle_count INTEGER DEFAULT 0,
            electric_vehicle_count INTEGER DEFAULT 0,
            fall_count INTEGER DEFAULT 0,
            danger_zone_count INTEGER DEFAULT 0,
            corridor_parking_count INTEGER DEFAULT 0,
            fire_exit_parking_count INTEGER DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()

def add_user(username: str, password: str, role: str = "user") -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)',
                      (username, password, role))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return -1
    finally:
        conn.close()

def get_user(username: str) -> Dict[str, Any]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0],
            "username": row[1],
            "password": row[2],
            "role": row[3],
            "created_at": row[4]
        }
    return None

def add_event(event_type: str, message: str, camera_id: int = None, frame_path: str = None) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO events (type, message, camera_id, frame_path)
        VALUES (?, ?, ?, ?)
    ''', (event_type, message, camera_id, frame_path))
    conn.commit()
    event_id = cursor.lastrowid
    conn.close()
    
    update_daily_stats(event_type)
    
    return event_id

def get_events(start_time: str = None, end_time: str = None, event_type: str = None) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = 'SELECT * FROM events WHERE 1=1'
    params = []
    
    if start_time:
        query += ' AND timestamp >= ?'
        params.append(start_time)
    if end_time:
        query += ' AND timestamp <= ?'
        params.append(end_time)
    if event_type:
        query += ' AND type = ?'
        params.append(event_type)
    
    query += ' ORDER BY timestamp DESC'
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    return [{
        "id": row[0],
        "type": row[1],
        "message": row[2],
        "camera_id": row[3],
        "frame_path": row[4],
        "timestamp": row[5]
    } for row in rows]

def add_camera(name: str, url: str = None) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO cameras (name, url) VALUES (?, ?)', (name, url))
    conn.commit()
    camera_id = cursor.lastrowid
    conn.close()
    return camera_id

def get_cameras() -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM cameras')
    rows = cursor.fetchall()
    conn.close()
    
    return [{
        "id": row[0],
        "name": row[1],
        "url": row[2],
        "enabled": row[3],
        "created_at": row[4]
    } for row in rows]

def add_recording(camera_id: int, filename: str, start_time: datetime, end_time: datetime = None, duration: int = None) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO recordings (camera_id, filename, start_time, end_time, duration)
        VALUES (?, ?, ?, ?, ?)
    ''', (camera_id, filename, start_time.isoformat(), end_time.isoformat() if end_time else None, duration))
    conn.commit()
    recording_id = cursor.lastrowid
    conn.close()
    return recording_id

def get_recordings(camera_id: int = None, start_time: str = None, end_time: str = None) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = 'SELECT * FROM recordings WHERE 1=1'
    params = []
    
    if camera_id:
        query += ' AND camera_id = ?'
        params.append(camera_id)
    if start_time:
        query += ' AND start_time >= ?'
        params.append(start_time)
    if end_time:
        query += ' AND start_time <= ?'
        params.append(end_time)
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    return [{
        "id": row[0],
        "camera_id": row[1],
        "filename": row[2],
        "start_time": row[3],
        "end_time": row[4],
        "duration": row[5]
    } for row in rows]

def update_daily_stats(event_type: str):
    today = datetime.now().date().isoformat()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM detection_stats WHERE date = ?', (today,))
    row = cursor.fetchone()
    
    if not row:
        cursor.execute('''
            INSERT INTO detection_stats (date) VALUES (?)
        ''', (today,))
        conn.commit()
        cursor.execute('SELECT * FROM detection_stats WHERE date = ?', (today,))
        row = cursor.fetchone()
    
    update_fields = {
        "person": "person_count",
        "vehicle": "vehicle_count", 
        "electric_vehicle": "electric_vehicle_count",
        "fall": "fall_count",
        "danger_zone": "danger_zone_count",
        "corridor_parking": "corridor_parking_count",
        "fire_exit_parking": "fire_exit_parking_count"
    }
    
    field = None
    if event_type == "fall":
        field = update_fields["fall"]
    elif event_type == "danger_zone":
        field = update_fields["danger_zone"]
    elif event_type == "corridor_parking":
        field = update_fields["corridor_parking"]
    elif event_type == "fire_exit_parking":
        field = update_fields["fire_exit_parking"]
    
    if field:
        cursor.execute(f'UPDATE detection_stats SET {field} = {field} + 1 WHERE date = ?', (today,))
        conn.commit()
    
    conn.close()

def get_daily_stats(date: str = None) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = 'SELECT * FROM detection_stats'
    params = []
    
    if date:
        query += ' WHERE date = ?'
        params.append(date)
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    return [{
        "id": row[0],
        "date": row[1],
        "person_count": row[2],
        "vehicle_count": row[3],
        "electric_vehicle_count": row[4],
        "fall_count": row[5],
        "danger_zone_count": row[6],
        "corridor_parking_count": row[7],
        "fire_exit_parking_count": row[8]
    } for row in rows]

init_db()

if not get_user("admin"):
    add_user("admin", hash_password("admin123"), "admin")