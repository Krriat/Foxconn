#!/usr/bin/env python3
"""Receive one complete JPEG per UDP datagram; ROS is not required.

On the receiving Linux desktop:
    sudo apt install python3-opencv python3-numpy
    python3 receive_udp_video.py
"""

import argparse
import select
import socket
import time

import cv2
import numpy as np


CAMERA_NAMES = {6000: "head", 6001: "left_wrist", 6002: "right_wrist"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="0.0.0.0", help="Local IPv4 bind address")
    parser.add_argument("--ports", nargs="+", type=int, default=[6000, 6002])
    parser.add_argument("--no-display", action="store_true", help="Print reception statistics without opening windows")
    args = parser.parse_args()
    if len(set(args.ports)) != len(args.ports) or any(
        port < 1 or port > 65535 for port in args.ports
    ):
        parser.error("Ports must be unique integers between 1 and 65535")

    sockets = {}
    counts = dict.fromkeys(args.ports, 0)
    windows = set()
    last_report = time.monotonic()
    try:
        for port in args.ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sockets[sock] = port
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
            sock.bind((args.bind, port))
            print(f"Listening on {args.bind}:{port} ({CAMERA_NAMES.get(port, 'camera')})", flush=True)
        print("Waiting for JPEG frames. Press q, Esc, or Ctrl+C to stop.", flush=True)

        while True:
            ready, _, _ = select.select(list(sockets), [], [], 0.02)
            for sock in ready:
                packet, sender = sock.recvfrom(65535)
                if not packet:
                    continue
                try:
                    frame = cv2.imdecode(np.frombuffer(packet, dtype=np.uint8), cv2.IMREAD_COLOR)
                except cv2.error:
                    continue
                if frame is None:
                    continue
                port = sockets[sock]
                if counts[port] == 0:
                    print(f"First frame: port={port}, sender={sender[0]}, size={frame.shape[1]}x{frame.shape[0]}", flush=True)
                counts[port] += 1
                if not args.no_display:
                    title = f"{CAMERA_NAMES.get(port, 'camera')} - UDP {port}"
                    if title not in windows:
                        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(title, 800, 450)
                        windows.add(title)
                    cv2.imshow(title, frame)

            if not args.no_display and cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            now = time.monotonic()
            if now - last_report >= 2:
                print("Received frames: " + ", ".join(f"{port}={count}" for port, count in counts.items()), flush=True)
                last_report = now
    except KeyboardInterrupt:
        pass
    except (OSError, cv2.error) as exc:
        print(f"Receiver failed: {exc}", flush=True)
        return 1
    finally:
        for sock in sockets:
            sock.close()
        if windows:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
