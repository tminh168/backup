from multiprocessing import Pool
from multiprocessing import cpu_count
from multi_counter import counter
import argparse

# check to see if this is the main thread of execution
if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--procs", type=int, default=-1,
		help="# of processes to spin up")
	args = vars(ap.parse_args())

	procs = args["procs"] if args["procs"] > 0 else cpu_count()

	args_counter = [
		('rtsp://192.168.200.78:556/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream', 'cam_1'),
		('rtsp://192.168.200.79:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream', 'cam_2'),
	]

	# construct and launch the processing pool
	print("[INFO] launching pool using {} processes...".format(procs))
	pool = Pool(processes=2)
	result = [pool.apply_async(counter, args=(k, v,)) for (i, (k, v)) in enumerate(args_counter)]

	# close the pool and wait for all processes to finish
	print("[INFO] waiting for processes to finish...")
	pool.close()
	pool.join()

	print("[INFO] multiprocessing complete")

