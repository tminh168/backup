#import multiprocessing as mp
from multiprocessing import Pool
from multiprocessing import cpu_count
from inference import tf_inference

def main():
    default_model = 'detection_1_edgetpu.tflite'
    secondary_model = 'detection_2_edgetpu.tflite'

    procs = cpu_count()
    print('Loading {} with {} labels.'.format('detection_edgetpu.tflite', 'people_label.txt'))
#    interpreter_1 = tflite.Interpreter(args.model, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
#    interpreter_2 = tflite.Interpreter(secondary_model, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
#    interpreter_1.allocate_tensors()
#    interpreter_2.allocate_tensors()
    args_counter = [
           ('rtsp://192.168.200.78:556/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream', default_model, './detected1'),
           ('rtsp://192.168.200.79:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream', secondary_model, './detected2')
    ]

    print("[INFO] launching pool using {} processes...".format(procs))
    pool = Pool(processes=procs)
    result = [pool.apply_async(tf_inference, args=(k, u, v,)) for (i, (k, u, v)) in enumerate(args_counter)]
    #processes = [mp.Process(target=tf_inference, args=(k, u, v,)) for (i, (k, u, v)) in enumerate(args_counter)]
    #for p in processes:
    #    p.start()

    print("[INFO] waiting for processes to finish...")
    pool.close()
    pool.join()
    #for p in processes:
     #   p.join()

    print("[INFO] multiprocessing complete")

if __name__ == '__main__':
    main()
