#enconding=utf-8
from playsound import playsound
import threading

class mp3player(object):
    def __init__(self):
        self.__bstart=True
        self.autoEvent = threading.Event()
        self.file_list=[]
        threading.Thread(target=self.__start).start()

    def __start(self):
        self.__bstart=True
        while(self.__bstart):
            self.autoEvent.wait()
            self.autoEvent.clear()
            if not self.__bstart:
                break
            if len(self.file_list)>0:
                self.start_play(self.file_list)

    def start_play(self,file_list):
            playsound('./voice/ping.mp3')
            playsound('./voice/{}.mp3'.format(file_list[0]))
            playsound('./voice/bei.mp3')
            playsound('./voice/{}.mp3'.format(file_list[1]))
            playsound('./voice/ren.mp3')
            playsound('./voice/{}.mp3'.format(file_list[2]))
    def play(self,file_list):
        self.file_list=file_list
        self.autoEvent.set()
    def stop(self):
        self.__bstart=False
        self.autoEvent.set()



