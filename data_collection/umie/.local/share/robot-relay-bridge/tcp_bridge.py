"""Private wired TCP bridge. Forwards bytes once; generates no relay commands."""
import argparse
import asyncio
import json
import signal
import socket
import time


class Bridge:
    def __init__(self, upstream_host, upstream_port, allowed_peer, upstream_source=None):
        self.upstream_host = upstream_host
        self.upstream_port = upstream_port
        self.allowed_peer = allowed_peer
        self.upstream_source = upstream_source
        self.clients = set()

    async def handle(self, reader, writer):
        peer = writer.get_extra_info('peername')
        task = asyncio.current_task()
        upstream = None
        pumps = []
        sent = [0, 0]
        if not peer or peer[0] != self.allowed_peer or len(self.clients) >= 4:
            writer.close()
            await writer.wait_closed()
            return
        self.clients.add(task)
        try:
            source = (self.upstream_source, 0) if self.upstream_source else None
            backend, upstream = await asyncio.wait_for(
                asyncio.open_connection(self.upstream_host, self.upstream_port, local_addr=source), .5)
            for stream in (writer, upstream):
                stream.get_extra_info('socket').setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            async def pump(source_reader, target_writer, index):
                while True:
                    data = await source_reader.read(65536)
                    if not data:
                        return
                    target_writer.write(data)
                    await asyncio.wait_for(target_writer.drain(), 1.)
                    sent[index] += len(data)

            pumps = [asyncio.create_task(pump(reader, upstream, 0)),
                     asyncio.create_task(pump(backend, writer, 1))]
            done, _ = await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
            for finished in done:
                finished.result()
        except (OSError, asyncio.TimeoutError) as error:
            print(json.dumps(dict(time_ns=time.time_ns(), event='connection_error',
                                  peer=peer[0], error=str(error))), flush=True)
        finally:
            for pump_task in pumps:
                pump_task.cancel()
            if pumps:
                await asyncio.gather(*pumps, return_exceptions=True)
            for stream in (upstream, writer):
                if stream:
                    stream.close()
            for stream in (upstream, writer):
                if stream:
                    try:
                        await stream.wait_closed()
                    except OSError:
                        pass
            self.clients.discard(task)
            print(json.dumps(dict(time_ns=time.time_ns(), event='connection_closed',
                                  peer=peer[0], forwarded_request_bytes=sent[0],
                                  forwarded_response_bytes=sent[1])), flush=True)

    async def close(self):
        clients = list(self.clients)
        for task in clients:
            task.cancel()
        if clients:
            await asyncio.gather(*clients, return_exceptions=True)


async def main(args):
    bridge = Bridge(args.upstream_host, args.upstream_port, args.allowed_peer, args.upstream_source)
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stopped.set)
    server = await asyncio.start_server(bridge.handle, args.listen_host, args.listen_port, backlog=4)
    print(json.dumps(dict(event='listening', host=args.listen_host, port=args.listen_port,
                          allowed_peer=args.allowed_peer, upstream=args.upstream_host,
                          upstream_source=args.upstream_source, command_generation=False,
                          retry_or_replay=False)), flush=True)
    try:
        await stopped.wait()
    finally:
        server.close()
        await server.wait_closed()
        await bridge.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--listen-host', required=True)
    parser.add_argument('--listen-port', type=int, default=20108)
    parser.add_argument('--allowed-peer', required=True)
    parser.add_argument('--upstream-host', required=True)
    parser.add_argument('--upstream-port', type=int, default=20108)
    parser.add_argument('--upstream-source', required=True)
    asyncio.run(main(parser.parse_args()))
