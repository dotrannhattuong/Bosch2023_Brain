# OD

import threading
import time
from jetson_od import Jet_OD

if __name__ =="__main__":
    jet_od = Jet_OD()
    t1 = threading.Thread(target=jet_od._rec)
    t2 = threading.Thread(target=jet_od._process)
    
    t1.start()
    t2.start()
    

    t1.join()
    t2.join()
    print("Done!")
