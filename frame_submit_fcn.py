# import the necessary packages
import time
import calendar
import subprocess
import requests
from requests.exceptions import ConnectionError


def frameSubmit(q):
    # initialize parameters
    url_tmp = 'http://ai-camera.dfm-europe.com/api/v1/admin/public/device-info'
    url_img = 'http://ai-camera.dfm-europe.com/api/v1/admin/public/uplink'
    t_init = time.time()
    run_flag = True

    while run_flag:
        t_check = time.time()
        if t_check >= t_init + 60:
            out = subprocess.Popen(['cat', '/sys/class/thermal/thermal_zone0/temp'],
                                   stdout=subprocess.PIPE).communicate()[0]

            temp = out.decode("utf-8").split('000')[0]
            print('temp: {}'.format(temp))
            current_time = calendar.timegm(time.gmtime())
            data = {
                "bus": 1,
                "temperature": temp,
                "timestamp": current_time
            }
            try:
                r = requests.post(url_tmp, data=data, verify=False)

                # check API response
                if r.status_code == 200:
                    print("Success")

            except ConnectionError:
                print("Check internet connection.")

            t_init = time.time()

        if not q.empty():
            t_send = time.time()
            data = q.get()
            try:
                r = requests.post(url_img, data=data, verify=False)

                # check API response
                if r.status_code == 200:
                    print("Success")
                else:
                    q.put(data)

            except ConnectionError:
                print("Check internet connection. Detection frame on standby!")
                q.put(data)

            print('Send: {}'.format(time.time() - t_send))

    return
