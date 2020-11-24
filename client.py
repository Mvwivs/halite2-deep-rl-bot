import asyncio

async def game(sockpath):
    game_map = {}
    reader, writer = await asyncio.open_unix_connection(sockpath)
    while True:
    
        game_map = await reader.readline()
        print(f'got game map {game_map}')
        
        writer.write('I SEND U ACTION\n'.encode())
        await writer.drain()

asyncio.run(game('bot.sock'))
