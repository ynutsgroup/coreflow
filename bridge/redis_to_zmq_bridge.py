from dotenv import load_dotenv
load_dotenv('/opt/coreflow/config/.env.bridge')

import redis, os, zmq, time, logging

r = redis.StrictRedis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind(f"tcp://{os.getenv('ZMQ_BIND')}:{os.getenv('ZMQ_PORT')}")
