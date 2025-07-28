import zmq
import json

def send_zmq_signal(zmq_cfg, signal):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.connect(zmq_cfg["ZMQ_ENDPOINT"])
    topic = zmq_cfg.get("ZMQ_TOPIC", "coreflow")
    socket.send_string(f"{topic} {json.dumps(signal)}")
    socket.close()
    context.term()
