#!/usr/bin/env python3
from argparse import *
import configparser
from utils import *
from numpy import array, sin, cos, arccos, dot
from numpy.linalg import norm
from render import Render
from framebuild import Frame
from collections import namedtuple
from sys import stderr
from pdb import set_trace as brk

def rot(theta):
	"""2D rotation matrix from an angle in radians"""
	c = cos(theta)
	s = sin(theta)
	return array(((c, -s), (s, c)))

UPPERARM, LOWERARM, UPPERLEG, LOWERLEG, TORSO, HEAD = range(6)

# For each body part, its centre of mass as a proportion of its height,
# assuming it's dangling vertically down, and what fraction of the total mass
# of the human body it is.
PartData = namedtuple("PartData", "com mass")
BODY_PARTS = {
		UPPERARM: PartData(0.53, 0.053),
		LOWERARM: PartData(0.57, 0.0504),
		UPPERLEG: PartData(0.56, 0.2444),
		LOWERLEG: PartData(0.57, 0.1222),
		TORSO: PartData(0.56, 0.463),
		HEAD: PartData(0.6, 0.0672),
		}

class BodyPart:
	def __init__(self, part, start, end):
		"""part is one of the BODY_PARTS above. start and end are its actual
		positions in 2D, projected into the plane of the bicycle"""
		self.part_data = BODY_PARTS[part]
		self.start = start
		self.end = end

	def com(self):
		"""Return the centre of mass in 2D space"""
		return self.start + (self.end - self.start) * self.part_data.com

	def mass(self):
		"""Return mass (as a proportion of the whole body mass)"""
		return self.part_data.mass

def com(body_parts):
	"""The centre of mass of a bunch of body parts"""
	ret = array([0., 0.])
	for bp in body_parts:
		ret += bp.com() * bp.mass()
	return ret

class Rider:
	def __init__(self, config):
		rider = self.rider_config = config["Rider"]
		bike = self.bike_config = config["Bike"]

		self.bar_width = float(bike["bar width"])
		self.shoulder_width = float(rider["shoulder width"])

		if rider["inseam"] == "unset":
			seat_height = float(bike["seat height"])
			inseam = seat_height / 0.883
		else:
			inseam = float(rider["inseam"])
			seat_height = inseam * 0.883

		# Origin is at BB of course. Work out position of hip
		angle = deg2rad(180 - float(bike["seat angle"]))
		self.hip = hip = array([cos(angle), sin(angle)]) * seat_height

		# Work out the length from the hip to the wrist assuming the angles we
		# want at the shoulder and elbow.
		snh = float(rider["sternal notch height"])

		# Working with a horizontal back for now-- we're just trying to find
		# the length and don't know the actual final angle.
		back = array([snh - inseam, 0])
		arm_len = self._arm_len("arm")
		forearm_len = self._arm_len("forearm")
		upper_arm_len = arm_len - forearm_len

		sa = deg2rad(180 - float(rider["shoulder angle"]))
		upper_arm = array([cos(sa), -sin(sa)]) * upper_arm_len
		ea = sa - deg2rad(float(rider["elbow angle"]))
		forearm = array([cos(ea), -sin(ea)]) * forearm_len

		hip_wrist = back + upper_arm + forearm
		hip_wrist_len = norm(hip_wrist)

		bar_drop = float(bike["bar drop"])
		x = sqrt(hip_wrist_len**2 - bar_drop**2)
		self.wrist = wrist = hip + array([x, -bar_drop])

		# Now that we know the actual position of the wrist, we can find the
		# proper angle of the back.
		v = wrist - hip
		v /= norm(v)
		w = hip_wrist / norm(hip_wrist)
		theta = arccos(dot(v, w))
		r = rot(theta)

		self.elbow = wrist - dot(r, forearm)
		self.shoulder = self.elbow - dot(r, upper_arm)

		offset = array([float(bike["bar reach"]), float(bike["bar stack"])])
		self.top_of_ht = wrist - offset

		try:
			height = float(rider["height"])
		except KeyError:
			height = snh + 300
			print(f"Warning: rider height not provided. Estimating {height}mm",
				file=stderr)
		self.head_len = height - snh

		# Now work out where the top of the head goes
		back = self.shoulder - self.hip
		back /= norm(back)
		self.top = self.shoulder + back * self.head_len

		self.com = self._find_com()

	def _arm_len(self, attr):
		"""Return an arm length in the plane of the bicycle"""
		offset = abs(self.shoulder_width - self.bar_width) / 2
		ret = float(self.rider_config[attr])
		if offset:
			ret = sqrt(ret**2 - offset**2)
		return ret

	def _find_com(self):
		foot = array([0., 0.])
		knee = self.hip/2

		parts = []
		for part, start, end in (
				(LOWERLEG, foot, knee),
				(UPPERLEG, knee, self.hip),
				(TORSO, self.hip, self.shoulder),
				(UPPERARM, self.elbow, self.shoulder),
				(LOWERARM, self.wrist, self.elbow),
				(HEAD, self.shoulder, self.top),
				):
			parts.append(BodyPart(part, start, end))

		return com(parts)

	def display(self, frame):
		stack, reach = self.top_of_ht[1], self.top_of_ht[0]
		print("You want a stack of {:.2f}mm and a reach of {:.2f}mm".format(
			stack, reach))

		back = self.shoulder - self.hip
		back /= norm(back)
		angle = rad2deg(arccos(dot(back, array([1, 0]))))
		print("Your back will be at an angle of {:.2f} deg "
				"from horizontal".format(angle))

		com_pos = self.com[0]
		if com_pos > 0:
			text, pos = "in front of", com_pos
		else:
			text, pos = "behind", -com_pos
		print(f"Your centre of mass will be {pos:.2f}mm "
			f"{text} the bottom bracket")

		if frame:
			fw, rw = frame.front_wheel_pos()[0], frame.rear_wheel_pos()[0]
			com_rel = com_pos - rw	# com relative to rear wheel.
			ratio = com_rel / (fw - rw)
			front_load = ratio * 100
			rear_load = (1-ratio) * 100
			print(f"Your front wheel will be loaded with {front_load:.2f}% and "
				f"your rear wheel with {rear_load:.2f}% of the rider weight")


	def draw(self, px_per_mm):
		inf = array([0, 0])

		points = [inf.copy()]
		sup = inf.copy()
		for attr in ("hip", "shoulder", "elbow", "wrist", "top_of_ht"):
			point = getattr(self, attr)
			points.append(point)
			for i in range(2):
				sup[i] = max(sup[i], point[i])
				inf[i] = min(inf[i], point[i])

		render = Render(inf, sup, px_per_mm)
		render.polyline(points, "black")

		render.circle(self.com, 20, "red")
		render.polyline([self.com, array([self.com[0], 0])], "red")

		return render

def main():
	ap = ArgumentParser()
	ap.add_argument("-f", "--frame_config", type=str, required=False,
			help="if supplied renders the frame under the rider")
	ap.add_argument("-r", "--resolution", type=int, default=2)
	ap.add_argument("config", type=str, nargs=1)

	args = ap.parse_args()

	config = configparser.ConfigParser()
	config.read(args.config)
	rider = Rider(config)

	render = rider.draw(args.resolution)

	frame = None
	if args.frame_config:
		frame_config = configparser.ConfigParser()
		frame_config.read(args.frame_config)
		frame = Frame(frame_config)
		render = frame.render_scale_diagram(render)

	rider.display(frame)
	render.save("rider.png")
	print("Rendered to rider.png")

if __name__ == "__main__":
	main()
