import io
import subprocess

import cv2

softsub_video = cv2.VideoCapture('softsub.mkv')
hardsub_video = cv2.VideoCapture('hardsub.mp4')
hardsub_lang = "fra"

softsub_fps = softsub_video.get(cv2.CAP_PROP_FPS)
hardsub_fps = hardsub_video.get(cv2.CAP_PROP_FPS)

def median(numbers):
	numbers = sorted(numbers)
	center = len(numbers) // 2
	if len(numbers) % 2 == 0:
		return sum(numbers[center - 1:center + 1]) / 2.0
	else:
		return numbers[center]

def frame_diff(a, b):
	diff = cv2.subtract(a, b)
	#diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
	height, width = diff.shape[:2]
	return sum(sum(cv2.reduce(diff, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S))[0]) / (height * width)

def find_key_frames(video, threshold=70, nbr_frames=5):
	key_frames = {}

	fps = video.get(cv2.CAP_PROP_FPS)
	#resolution = video.get(cv2.CAP_PROP_FRAME_WIDTH) * video.get(cv2.CAP_PROP_FRAME_HEIGHT)

	ok, last = video.read()
	if not ok:
		return key_frames
	while(len(key_frames) < nbr_frames):
		ok, current = video.read()
		if not ok:
			break
		#diff = cv2.subtract(last, current)
		#cv2.imshow("last", last)
		#cv2.imshow("current", current)
		#cv2.imshow("diff", diff)
		d = frame_diff(last, current)
		if d > threshold:
			pos = video.get(cv2.CAP_PROP_POS_FRAMES)
			key_frames[pos] = current
			print("Found key frame:", pos, d)
			#cv2.waitKey(0)
		#else:
		#	cv2.waitKey(int(1/fps*1000))
		last = current

	return key_frames

def match_keyframes(softsub_frames, hardsub_frames, max_diff=10):
	matches = []
	for softsub_pos, softsub_frame in softsub_frames.items():
		best_diff = float("inf")
		best_frame = None
		best_pos = -1

		for hardsub_pos, hardsub_frame in hardsub_frames.items():
			d = frame_diff(softsub_frame, hardsub_frame)
			if d > max_diff:
				continue
			if d < best_diff:
				best_diff = d
				best_frame = hardsub_frame
				best_pos = hardsub_pos

		if best_frame is None:
			continue

		pos_diff_sec = softsub_pos/softsub_fps - best_pos/hardsub_fps
		print("image_diff={} pos_diff_sec={}".format(best_diff, pos_diff_sec))
		#cv2.imshow("softsub", softsub_frame)
		#cv2.imshow("hardsub", best_frame)
		#cv2.waitKey(0)
		matches.append(pos_diff_sec)
	return median(matches)

initial_pos = 24 * 20

# Align videos
softsub_video.set(cv2.CAP_PROP_POS_FRAMES, initial_pos)
hardsub_video.set(cv2.CAP_PROP_POS_FRAMES, initial_pos)
softsub_key_frames = find_key_frames(softsub_video)
hardsub_key_frames = find_key_frames(hardsub_video)
softsub_video.set(cv2.CAP_PROP_POS_FRAMES, initial_pos)
hardsub_video.set(cv2.CAP_PROP_POS_FRAMES, initial_pos)

# pos_diff_sec = softsub_pos - hardsub_pos
pos_diff_sec = match_keyframes(softsub_key_frames, hardsub_key_frames)
print("pos_diff_sec={}".format(pos_diff_sec))

while(True):
	softsub_pos = softsub_video.get(cv2.CAP_PROP_POS_FRAMES)
	softsub_t = softsub_pos/softsub_fps

	ok, softsub_frame = softsub_video.read()
	if not ok:
		break

	# TODO: this works in one way only
	hardsub_frame = None
	while(True):
		hardsub_pos = hardsub_video.get(cv2.CAP_PROP_POS_FRAMES)
		hardsub_t = hardsub_pos/hardsub_fps

		ok, hardsub_frame = hardsub_video.read()
		if not ok:
			break

		if hardsub_t >= softsub_t - pos_diff_sec:
			break
	if hardsub_frame is None:
		break

	#diff = cv2.absdiff(softsub_frame, hardsub_frame)
	diff = cv2.subtract(255-softsub_frame, 255-hardsub_frame)
	diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
	#ok, diff = cv2.threshold(diff, 250, 255, cv2.THRESH_BINARY)
	ok, diff = cv2.threshold(diff, 254, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	diff = 255 - diff
	#diff = autocrop(diff, threshold=50)
	cv2.imshow('diff', diff)

	key = cv2.waitKey(int(1/softsub_fps*1000))
	if key == ord(' '):
		cv2.waitKey(0)
	if key == ord('q'):
		break
	if key == ord('\n'):
		cv2.imwrite("output.png", diff)
		ok, buf = cv2.imencode(".bmp", diff)
		# TODO: error handling
		p = subprocess.Popen(['/usr/bin/tesseract', 'stdin', 'stdout', '-l', hardsub_lang], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
		p.stdin.write(buf)
		p.stdin.close()
		for line in io.TextIOWrapper(p.stdout, encoding="utf-8"):
			line = line.rstrip()
			if len(line) == 0:
				continue
			print(line)
		p.wait()

softsub_video.release()
hardsub_video.release()

cv2.destroyAllWindows()
