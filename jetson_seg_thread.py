# Seg
import threading
import time
from jetson_seg import Jet_Seg

if __name__ =="__main__":
    jet_seg = Jet_Seg()

    t1 = threading.Thread(target=jet_seg._send)
    t2 = threading.Thread(target=jet_seg._rec)
    t3 = threading.Thread(target=jet_seg._process)

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()
    print("Done!")
