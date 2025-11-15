#!/usr/bin/env python3
from argparse import *
import configparser
from utils import *
from numpy import array, sin, cos, arccos, dot
from numpy.linalg import norm
from render import Render
from framebuild import Frame
from pdb import set_trace as brk

def rot(theta):
	"""2D rotation matrix from an angle in radians"""
	c = cos(theta)
	s = sin(theta)
	return array(((c, -s), (s, c)))

#Class for eacht bodypart.
class Limb:
	def __init__(
		self,
		start,
		stop,
		weight,
		center_off_mass,
		):
		self.length = sqrt((start[0]-stop[0])**2 + (start[1]-stop[1])**2)
		self.weight = weight
		self.rel_com_pos = center_off_mass

		self.pos_start = start
		self.pos_stop = stop
		self.com_pos = self.get_com()

	def get_com(self):
		dist_x = (self.pos_stop[0] - self.pos_start[0]) * self.rel_com_pos
		dist_y = (self.pos_stop[1] - self.pos_start[1]) * self.rel_com_pos 

		return (self.pos_start[0]+dist_x, self.pos_start[1]+dist_y)


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
		self.center_of_mass_pos = self.center_of_mass()

	# Calculates the riders center of mass / center of gravity. It calculates the weight of each bodypart, 
	# together with each bodyparts center of mass. Afterwards the center of mass of these points is calculated, 
	# wich is the com if the rider.
	def center_of_mass(
		self,
		# com_pos_weight = weight distribution within the body. 
		# Center of mass position is measured from ground up, when standing upright. 
		# Tuple(center of mass position, weight proportional to complete body 
		# (e.g. 0.3 = 30% of the bodys weight is in this in this bodypart))
		com_pos_weight = { 
				"upperarm" : (0.53,0.053),
				"lowerarm" : (0.57,0.0504),
				"upperleg" : (0.56,0.2444),
				"lowerleg" : (0.57,0.1222),
				"torso" : (0.56,0.463),
				"head" : (0.6,0.0672),
			},
		weight=80. #irrelevant atm. Maybe useful, if the bikes weight is used as well?
		):
		foot = array([0, 0])
		knee = array([self.hip[0]/2, self.hip[1]/2])
		lowerleg = Limb(foot, knee, com_pos_weight["lowerleg"][1]*weight, com_pos_weight["lowerleg"][0])
		upperleg = Limb(knee, self.hip, com_pos_weight["upperleg"][1]*weight, com_pos_weight["upperleg"][0])
		torso = Limb(self.hip, self.shoulder, com_pos_weight["torso"][1]*weight, com_pos_weight["torso"][0])
		upperarm = Limb(self.elbow, self.shoulder, com_pos_weight["upperarm"][1]*weight, com_pos_weight["upperarm"][0])
		lowerarm = Limb(self.wrist, self.elbow, com_pos_weight["lowerarm"][1]*weight, com_pos_weight["lowerarm"][0])
		head = Limb(self.shoulder, (self.shoulder[0], self.shoulder[1] + 30),com_pos_weight["head"][1]*weight, com_pos_weight["head"][0])

		com_x = 0
		com_y = 0
		mass = 0
		for bodypart in [lowerleg, upperleg, torso, upperarm, lowerarm, head]:
			com_x += (bodypart.weight * bodypart.com_pos[0])
			com_y += (bodypart.weight * bodypart.com_pos[1])
			mass += (bodypart.weight)

		return (com_x/mass, com_y/mass)

	def _arm_len(self, attr):
		"""Return an arm length in the plane of the bicycle"""
		offset = abs(self.shoulder_width - self.bar_width) / 2
		ret = float(self.rider_config[attr])
		if offset:
			ret = sqrt(ret**2 - offset**2)
		return ret

	def display(self, frame = None):
		stack, reach = self.top_of_ht[1], self.top_of_ht[0]
		print("You want a stack of {:.2f}mm and a reach of {:.2f}mm".format(
			stack, reach))

		back = self.shoulder - self.hip
		back /= norm(back)
		angle = rad2deg(arccos(dot(back, array([1, 0]))))
		print("Your back will be at an angle of {:.2f} deg "
				"from horizontal".format(angle))
		print(f"Your Center of Mass will be {self.center_of_mass_pos[0]:.2f}mm, relative to the bottom bracket")

		if frame is not None:
			rear_wheel = frame.left_cs.tube_top[0]
			#use the rear wheel position as a datum
			rear_wheel_distance = rear_wheel*-1
			wheelbase = frame._fork_path()[-1][0] + rear_wheel_distance
			
			com_pos = rear_wheel_distance + self.center_of_mass_pos[0]

			distribution = com_pos/wheelbase * 100
			distribution_rear = 100 - distribution 
			print(f"Your Front Wheel will be loaded {distribution:.2f}%, your rear Wheel {distribution_rear:.2f}")

	def draw(self, px_per_mm, frame = None):
		inf = array([0, 0])

		points = [inf.copy()]
		sup = inf.copy()
		for attr in ("hip", "shoulder", "elbow", "wrist", "top_of_ht"):
			point = getattr(self, attr)
			points.append(point)
			for i in range(2):
				sup[i] = max(sup[i], point[i])
				inf[i] = min(inf[i], point[i])

		if frame is not None:
			frame_inf = array([frame.left_cs.tube_top[0], 0])
			frame_sup = array([frame._fork_path()[-1][0], frame.head_tube.tube_top[1]])
			for i in range(2):
				inf[i] = min(inf[i], frame_inf[i])
				sup[i] = max(frame_sup[i], sup[i])

		render = Render(inf, sup, px_per_mm)
		render.polyline(points, "black")
		render.circle(self.center_of_mass_pos, 20, "red")
		com_line = [self.center_of_mass_pos, (self.center_of_mass_pos[0], inf[1])]
		render.polyline(com_line, "red")

		if frame is not None:
			render = frame.render_scale_diagram(render)

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


	if args.frame_config:
		frame_config = configparser.ConfigParser()
		frame_config.read(args.frame_config)
		frame = Frame(frame_config)
	else:
		frame = None
	
	rider.display(frame)
	render = rider.draw(args.resolution, frame)

	if frame is not None:
		render = frame.render_scale_diagram(render)

	render.save("rider.png")
	print("Rendered to rider.png")

if __name__ == "__main__":
	main()
