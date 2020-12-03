from python:3.6-buster

WORKDIR ml-starter-bot 
COPY requirements.txt . 
RUN pip3 install -r requirements.txt --no-cache-dir
COPY . .
CMD ["make"]
# CMD ["./halite", "-i", "replays", "-d", "240 160", "'python3 FakeBot.py'", "'python3 Enemy.py'"]
