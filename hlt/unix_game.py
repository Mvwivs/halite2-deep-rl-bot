import sys
import logging
import copy
import socket

from . import game_map

class GameUnix:
    """
    :ivar map: Current map representation
    :ivar initial_map: The initial version of the map before game starts
    """

    def _send_string(self, s):
        """
        Send data to the game. Call :function:`done_sending` once finished.

        :param str s: String to send
        :return: nothing
        """
        self.sfile.write(s)

    def _done_sending(self):
        """
        Finish sending commands to the game.

        :return: nothing
        """
        self.sfile.write('\n')
        self.sfile.flush()

    def _get_string(self):
        """
        Read input from the game.

        :return: The input read from the Halite engine
        :rtype: str
        """
        result = self.sfile.readline().rstrip('\n')
        return result

    def send_command_queue(self, command_queue):
        """
        Issue the given list of commands.

        :param list[str] command_queue: List of commands to send the Halite engine
        :return: nothing
        """
        for command in command_queue:
            self._send_string(command)

        self._done_sending()

    @staticmethod
    def _set_up_logging(tag, name):
        """
        Set up and truncate the log

        :param tag: The user tag (used for naming the log)
        :param name: The bot name (used for naming the log)
        :return: nothing
        """
        log_file = "{}_{}.log".format(tag, name)
        logging.basicConfig(filename=log_file, level=logging.DEBUG, filemode='w')
        logging.info("Initialized bot {}".format(name))

    def __init__(self, name, socket_path="/dev/shm/bot.sock"):
        """
        Initialize the bot with the given name.

        :param name: The name of the bot.
        """
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connected = False
        while not connected:
            try:
                self.s.connect(socket_path)
                connected = True
            except Exception:
                pass    # Do nothing, just try again
        self.sfile = self.s.makefile('rw')

        self._name = name
        self._send_name = False
        tag = int(self._get_string())
        GameUnix._set_up_logging(tag, name)
        width, height = [int(x) for x in self._get_string().strip().split()]
        self.map = game_map.Map(tag, width, height)
        self.update_map()
        self.initial_map = copy.deepcopy(self.map)
        self._send_name = True

        self.done = False

    def update_map(self):
        """
        Parse the map given by the engine.

        :return: new parsed map
        :rtype: game_map.Map
        """
        if self._send_name:
            self._send_string(self._name)
            self._done_sending()
            self._send_name = False
        logging.info("---NEW TURN---")
        recv = self._get_string()

        if recv == "":
            self.close()
            self.done = True
            return self.map     # last step map

        self.map._parse(recv)
        return self.map
    
    def close(self):
        self.sfile.close()
        self.s.close()

class GameStdIO:
    """
    :ivar map: Current map representation
    :ivar initial_map: The initial version of the map before game starts
    """

    def _send_string(self, s):
        """
        Send data to the game. Call :function:`done_sending` once finished.

        :param str s: String to send
        :return: nothing
        """
        sys.stdout.write(s)

    def _done_sending(self):
        """
        Finish sending commands to the game.

        :return: nothing
        """
        sys.stdout.write('\n')
        sys.stdout.flush()

    def _get_string(self):
        """
        Read input from the game.

        :return: The input read from the Halite engine
        :rtype: str
        """
        result = sys.stdin.readline().rstrip('\n')
        return result

    def send_command_queue(self, command_queue):
        """
        Issue the given list of commands.

        :param list[str] command_queue: List of commands to send the Halite engine
        :return: nothing
        """
        for command in command_queue:
            self._send_string(command)

        self._done_sending()

    @staticmethod
    def _set_up_logging(tag, name):
        """
        Set up and truncate the log

        :param tag: The user tag (used for naming the log)
        :param name: The bot name (used for naming the log)
        :return: nothing
        """
        log_file = "{}_{}.log".format(tag, name)
        logging.basicConfig(filename=log_file, level=logging.DEBUG, filemode='w')
        logging.info("Initialized bot {}".format(name))

    def __init__(self, name):
        """
        Initialize the bot with the given name.

        :param name: The name of the bot.
        """

        self._name = name
        self._send_name = False
        tag = int(self._get_string())
        GameStdIO._set_up_logging(tag, name)
        width, height = [int(x) for x in self._get_string().strip().split()]
        self.map = game_map.Map(tag, width, height)
        self.update_map()
        self.initial_map = copy.deepcopy(self.map)
        self._send_name = True

        self.done = False

    def update_map(self):
        """
        Parse the map given by the engine.

        :return: new parsed map
        :rtype: game_map.Map
        """
        if self._send_name:
            self._send_string(self._name)
            self._done_sending()
            self._send_name = False
        logging.info("---NEW TURN---")
        recv = self._get_string()

        if recv == "":
            self.close()
            self.done = True
            return self.map     # last step map

        self.map._parse(recv)
        return self.map
    
    def close(self):
        pass