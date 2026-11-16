"""Loopback mock relay only: no real hardware, no ROS, no actuator access."""
import asyncio
import unittest
from tcp_bridge import Bridge


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_fragmented_requests_forwarded_once_and_reply_unchanged(self):
        received = []
        request = bytes.fromhex('00010000000601050000ff00')

        async def mock(reader, writer):
            data = await reader.readexactly(12)
            received.append(data)
            for part in (data[:3], data[3:8], data[8:]):
                writer.write(part)
                await writer.drain()
                await asyncio.sleep(.005)
            received.append(await reader.read())
            writer.close()
            await writer.wait_closed()

        backend = await asyncio.start_server(mock, '127.0.0.1', 0)
        bridge = Bridge('127.0.0.1', backend.sockets[0].getsockname()[1], '127.0.0.1')
        server = await asyncio.start_server(bridge.handle, '127.0.0.1', 0)
        try:
            reader, writer = await asyncio.open_connection('127.0.0.1', server.sockets[0].getsockname()[1])
            for part in (request[:4], request[4:]):
                writer.write(part)
                await writer.drain()
                await asyncio.sleep(.005)
            self.assertEqual(await asyncio.wait_for(reader.readexactly(12), 1), request)
            writer.close()
            await writer.wait_closed()
            await asyncio.sleep(.05)
            self.assertEqual(received, [request, b''])
        finally:
            server.close(); backend.close()
            await server.wait_closed(); await backend.wait_closed(); await bridge.close()

    async def test_unavailable_backend_closes_client_without_retry(self):
        reservation = await asyncio.start_server(lambda r,w:None, '127.0.0.1', 0)
        port = reservation.sockets[0].getsockname()[1]
        reservation.close(); await reservation.wait_closed()
        bridge = Bridge('127.0.0.1', port, '127.0.0.1')
        server = await asyncio.start_server(bridge.handle, '127.0.0.1', 0)
        try:
            reader, writer = await asyncio.open_connection('127.0.0.1', server.sockets[0].getsockname()[1])
            self.assertEqual(await asyncio.wait_for(reader.read(), 1), b'')
            writer.close(); await writer.wait_closed()
        finally:
            server.close(); await server.wait_closed(); await bridge.close()

    async def test_unapproved_peer_never_connects_to_backend(self):
        accepted = []
        async def mock(reader, writer):
            accepted.append(True); writer.close(); await writer.wait_closed()
        backend = await asyncio.start_server(mock, '127.0.0.1', 0)
        bridge = Bridge('127.0.0.1', backend.sockets[0].getsockname()[1], '192.0.2.1')
        server = await asyncio.start_server(bridge.handle, '127.0.0.1', 0)
        try:
            reader, writer = await asyncio.open_connection('127.0.0.1', server.sockets[0].getsockname()[1])
            self.assertEqual(await asyncio.wait_for(reader.read(), 1), b'')
            writer.close(); await writer.wait_closed()
            self.assertEqual(accepted, [])
        finally:
            server.close(); backend.close()
            await server.wait_closed(); await backend.wait_closed(); await bridge.close()


if __name__ == '__main__':
    unittest.main()
