# import the necessary packages
import time
import calendar
import subprocess
import requests
from requests.exceptions import ConnectionError


def frameSubmit(q, check_temp):
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
            check_temp = temp
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
            continue

        if not q.empty():
            datas = []
            if q.qsize() >= 3:
                for i in range(3):
                    data = q.get()
                    datas.append(data)
            else:
                data = q.get()
                datas.append(data)

            try:
                r = requests.post(url_img, json=datas, verify=False)

                # check API response
                if r.status_code == 200:
                    print("Success")
                elif r.status_code == 429:
                    for data in datas:
                        q.put(data)
                    time.sleep(0.5)
                    print("Too many requests..")
                else:
                    for data in datas:
                        q.put(data)

            except ConnectionError:
                print("Check internet connection. Detection frame on standby!")
                for data in datas:
                    q.put(data)

    return
