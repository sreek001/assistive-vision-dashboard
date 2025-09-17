from flask import Flask, request, jsonify, render_template, send_from_directory
from datetime import datetime, timedelta
import base64
import os
import threading
import json
import logging
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")

store = {
    "latest_image_b64": None,
    "latest_detections": [],
    "latest_sensor": None,
    "latest_location": None,
    "latest_sos": None,
    "timestamp": None,
    "device_status": {},
    "alert_history": deque(maxlen=50),
    "stats": {
        "total_detections": 0,
        "sos_count": 0,
        "uptime_start": datetime.utcnow().isoformat()
    }
}
store_lock = threading.Lock()

def log_alert(alert_type, message, device_id=None):
    alert = {
        "type": alert_type,
        "message": message,
        "device_id": device_id,
        "timestamp": datetime.utcnow().isoformat()
    }
    with store_lock:
        store["alert_history"].append(alert)
    logger.info(f"ALERT [{alert_type}]: {message}")

def update_device_status(device_id, device_type="unknown"):
    with store_lock:
        store["device_status"][device_id] = {
            "type": device_type,
            "last_seen": datetime.utcnow().isoformat(),
            "status": "online"
        }

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/latest', methods=['GET'])
def latest():
    with store_lock:
        cutoff = datetime.utcnow() - timedelta(seconds=30)
        for device_id, status in store["device_status"].items():
            if "last_seen" in status:
                last_seen = datetime.fromisoformat(status["last_seen"])
                status["status"] = "online" if last_seen > cutoff else "offline"
        
        return jsonify({
            **store,
            "alert_history": list(store["alert_history"])[-10:]
        })

@app.route('/update', methods=['POST'])
def update():
    try:
        data = request.get_json(force=True)
        if not data or 'source' not in data:
            return jsonify({"error": "no data or missing source"}), 400

        ts = data.get("timestamp") or datetime.utcnow().isoformat()
        source = data.get('source')
        
        with store_lock:
            if source == 'vision':
                store['latest_image_b64'] = data.get('image')
                detections = data.get('detections', [])
                store['latest_detections'] = detections
                store['timestamp'] = ts
                store['stats']['total_detections'] += len(detections)
                update_device_status("vision_client", "laptop")
                important_objects = ['person', 'car', 'truck', 'bicycle', 'motorcycle']
                for det in detections:
                    if det.get('name') in important_objects and det.get('confidence', 0) > 0.7:
                        log_alert("DETECTION", f"High confidence {det['name']} detected", "vision_client")
                        
            elif source == 'esp':
                distance = data.get('distance')
                device_id = data.get('device_id', 'esp_unknown')
                sos = data.get('sos', False)
                
                store['latest_sensor'] = {
                    "distance": distance, 
                    "device_id": device_id,
                    "timestamp": ts
                }
                update_device_status(device_id, "esp8266")
                if sos:
                    store['latest_sos'] = {"device_id": device_id, "timestamp": ts}
                    store['stats']['sos_count'] += 1
                    log_alert("SOS", f"Emergency signal from {device_id}", device_id)
                if distance and 0 < distance < 50:
                    log_alert("OBSTACLE", f"Close obstacle detected: {distance}cm", device_id)
                store['timestamp'] = ts
            else:
                return jsonify({"error": "unknown source"}), 400

        return jsonify({"status": "ok", "timestamp": ts}), 200
    except Exception as e:
        logger.error(f"Update error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/location', methods=['POST'])
def location():
    try:
        data = request.get_json(force=True)
        lat = data.get('lat')
        lon = data.get('lon')
        who = data.get('who', 'unknown')
        ts = datetime.utcnow().isoformat()
        
        with store_lock:
            store['latest_location'] = {
                "lat": lat, 
                "lon": lon, 
                "who": who, 
                "timestamp": ts
            }
        
        log_alert("LOCATION", f"Location update from {who}", who)
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        logger.error(f"Location error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    with store_lock:
        return jsonify({
            "status": "healthy",
            "uptime": store['stats']['uptime_start'],
            "devices": store['device_status'],
            "stats": store['stats']
        })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
