import random

from paho.mqtt import client as mqtt_client
from crop import image_resize

import cv2
import matplotlib.image as mpimg
import io
import numpy as np


broker = '0.0.0.0'
port = 1883
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 100)}'
depth = None
image = None
# username = 'emqx'
# password = 'public'


def connect_mqtt() -> mqtt_client:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    # client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        print(f"Received `{len(msg.payload)}` from `{msg.topic}` topic")

        fp = io.BytesIO(msg.payload)

        with fp:
            img = mpimg.imread(fp, format='png')

        if msg.topic == "/depth":
            global depth
            depth = img
        elif msg.topic == "/image":
            global image
            image = img

        if depth is not None and image is not None:
            right = image_resize(depth, width=image.shape[1])
            cv2.imshow("Image", np.hstack([image, right]))
            cv2.waitKey(1)

    client.subscribe("/depth")
    client.subscribe("/image")
    client.on_message = on_message


def run():
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()


if __name__ == '__main__':
    run()