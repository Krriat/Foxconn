#!/usr/bin/env python3
import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RealSense live stream</title>
<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif;text-align:center}
h2{font-weight:500}img{max-width:96vw;max-height:85vh;border:1px solid #444}</style>
</head><body><h2>RealSense 摄像头实时画面</h2>
<img src="/stream.mjpg" alt="RealSense stream"></body></html>""".encode("utf-8")


class Camera:
    def __init__(self, device, fourcc):
        self.lock = threading.Lock()
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {device}")

    def jpeg(self):
        with self.lock:
            ok, frame = self.cap.read()
        if not ok:
            return None
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return encoded.tobytes() if ok else None


def make_handler(camera):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)
                return
            if self.path != "/stream.mjpg":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpeg = camera.jpeg()
                    if jpeg is None:
                        time.sleep(0.03)
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, fmt, *args):
            print(f"{self.client_address[0]} - {fmt % args}", flush=True)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/video6")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fourcc", default="UYVY")
    args = parser.parse_args()
    camera = Camera(args.device, args.fourcc)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(camera))
    print(f"RealSense live stream: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    finally:
        camera.cap.release()


if __name__ == "__main__":
    main()
