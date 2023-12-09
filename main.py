from VideoTerminal import VideoTerminal
from threading import Thread

term = VideoTerminal()
t1 = Thread(target=term.start_video_feed)

t1.start()
