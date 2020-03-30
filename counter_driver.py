from multiprocessing import Pool
from multiprocessing import cpu_count
from counter import count_ppl
import argparse

# check to see if this is the main thread of execution
if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--procs", type=int, default=-1,
		help="# of processes to spin up")
	args = vars(ap.parse_args())

	procs = args["procs"] if args["procs"] > 0 else cpu_count()

	args_counter = [
		("./input_video/8fps.mp4", "./detected_1"),
		("./input_video/15fps.mp4", "./detected_2"),
	]

	# construct and launch the processing pool
	print("[INFO] launching pool using {} processes...".format(procs))
	
	pool = Pool(processes=procs)
	result = [pool.apply_async(count_ppl, args=(k, v,)) for (i, (k, v)) in enumerate(args_counter)]

	# close the pool and wait for all processes to finish
	print("[INFO] waiting for processes to finish...")
	pool.close()
	pool.join()

	print("[INFO] multiprocessing complete")

