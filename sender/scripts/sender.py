# !/usr/bin/env python

import rospy
import websocket
import json

# from msg.ObjectArray import ObjectArray
from detection.msg._ObjectArray import ObjectArray

LABELS = ["human", "unknown", "animals"]

try:
    import thread
except ImportError:
    import _thread as thread
import time


def on_message(ws, message):
    pass
    # print(message)


def on_error(ws, error):
    print(error)


def on_close(ws):
    print("### closed ###")


def on_open(ws):
    def run(*args):
        drone_id = 1
        latitude = 55.751574
        longtitude = 37.573856
        while True:
            if on_object_array.last_object_array is None:
                continue
            object_array = on_object_array.last_object_array
            on_object_array.last_object_array = None
            cnt = {l: 0 for l in LABELS}
            for object in object_array.objects:
                cnt[LABELS[object.label]] += 1
            data = {
                'drone_id': drone_id,
                'latitude': latitude,
                'longtitude': longtitude,
                'timestamp': time.time(),
                'humans': cnt['human'],
                'animals': cnt["animals"],
                'unknown': cnt["unknown"]
            }
            time.sleep(1)
            latitude += 0.001
            ws.send(json.dumps(data))
        ws.close()

    thread.start_new_thread(run, ())


def on_object_array(msg):
    on_object_array.last_object_array = msg


on_object_array.last_object_array = None


if __name__ == "__main__":
    rospy.init_node("sender")

    sub_objects = rospy.Subscriber("/detection/yolo/objects", ObjectArray, on_object_array)

    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("ws://192.168.86.248:8080/drone/", on_message=on_message, on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()
