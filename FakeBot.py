
import asyncio 
import sys
import os
import signal
from pathlib import Path

async def write(writer, line):
    writer.write(line.encode())
    await writer.drain()

async def read(reader):
    got = await reader.readline()
    sys.stdout.write(got.decode('utf-8'))
    sys.stdout.flush()

async def bot(reader, writer):
    await write(writer, sys.stdin.readline())
    await write(writer, sys.stdin.readline())

    for line in sys.stdin:
        await write(writer, line)
        await read(reader)

async def launch(path):
    server = await asyncio.start_unix_server(bot, path=path)
    async with server:
        await server.serve_forever()
    os.unlink(path)

socket_path = sys.argv[1]
old_socket = Path(socket_path)
old_socket.unlink(missing_ok=True)
loop = asyncio.get_event_loop()
loop.add_signal_handler(signal.SIGTERM, lambda: os.unlink(socket_path))

asyncio.run(launch(socket_path))
