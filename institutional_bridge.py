#!/opt/coreflow/venv/bin/python
import zmq, asyncio, logging

async def bridge():
    ctx = zmq.Context()
    try:
        rx = ctx.socket(zmq.SUB)
        rx.bind("tcp://*:5555")
        rx.setsockopt_string(zmq.SUBSCRIBE, "")
        
        tx = ctx.socket(zmq.PUB)
        tx.connect("tcp://10.10.10.40:5555")
        
        logging.basicConfig(level=logging.INFO)
        logging.info("Bridge aktiviert")
        
        while True:
            msg = await rx.recv_string()
            await tx.send_string(msg)
            logging.debug(f"Weitergeleitet: {msg}")
            
    finally:
        ctx.term()

if __name__ == "__main__":
    try:
        asyncio.run(bridge())
    except KeyboardInterrupt:
        logging.info("Bridge gestoppt")
