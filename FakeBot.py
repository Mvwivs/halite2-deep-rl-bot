
import asyncio 
import sys

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

async def launch():
    server = await asyncio.start_unix_server(bot, path='bot.sock')
    async with server:
        await server.serve_forever()

asyncio.run(launch())
