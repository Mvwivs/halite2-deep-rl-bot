
import asyncio 
import sys


async def bot(reader, writer):
    with open('debug.log', 'w') as f:
        line = sys.stdin.readline()
        writer.write(line.encode())
        f.write(f'write: {line}')
        f.flush()
        await writer.drain()
        line = sys.stdin.readline()
        writer.write(line.encode())
        f.write(f'write: {line}')
        f.flush()
        await writer.drain()

        line = sys.stdin.readline()
        writer.write(line.encode())
        f.write(f'write: {line}')
        f.flush()
        await writer.drain()

        got = await reader.readline()
        sys.stdout.write(got.decode('utf-8'))
        sys.stdout.flush()
        # print('\n', flush=True)
        f.write(f'read: {got}\n')
        f.flush()

        for line in sys.stdin:
            writer.write(line.encode())
            f.write(f'write: {line}')
            f.flush()
            await writer.drain()
            got = await reader.readline()
            sys.stdout.write(got.decode('utf-8'))
            sys.stdout.flush()
            # print('\n', flush=True)
            f.write(f'read: {got}\n')
            f.flush()

async def launch():
    server = await asyncio.start_unix_server(bot, path='bot.sock')
    async with server:
        await server.serve_forever()

asyncio.run(launch())
