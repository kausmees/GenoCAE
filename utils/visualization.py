import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()
from matplotlib.colors import rgb2hex
from utils.data_handler import get_pop_superpop_list, get_superpop_pop_dict
import matplotlib.animation as animation
import gc


######################################## Plot settings ###################################
fsize = 3.3
markersize = 10
lw_scatter_points = 0.1
lw_figure = 0.01

#########################################

sns.set_style(style="whitegrid", rc=None)

#########################################


plt.rcParams.update({'xtick.labelsize': 5})
plt.rcParams.update({'ytick.labelsize': 5})
plt.rcParams.update({'font.size': 6})
plt.rcParams.update({'axes.titlesize': 4})
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams.update({'xtick.major.size': 2})
plt.rcParams.update({'ytick.major.size': 2})
plt.rcParams.update({'xtick.major.width': 0.0})
plt.rcParams.update({'ytick.major.width': 0.0})
plt.rcParams.update({'xtick.major.pad': 0.0})
plt.rcParams.update({'ytick.major.pad': 0.0})
plt.rcParams.update({'axes.labelpad': 1.5})
plt.rcParams.update({'axes.titlepad': 0})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'lines.linewidth': 0.5})
plt.rcParams.update({'grid.linewidth': 0.1})

#########################################################################################




def plot_coords(coords, outfileprefix, savefig = True):
	'''
	Plot the first two dimensions of the given data in a scatter plot.

	Returns data that can be used to generate animation.

	:param coords: the data to plot. array of (n_samples x n_dim), n_dim >= 2
	:param outfileprefix: directory and filename, without extension, to save plot to
	:param savefig: save plot to disk
	'''


	plt.figure(figsize=(fsize,fsize), linewidth = lw_figure)
	plt.scatter(coords[:,0], coords[:,1], color="gray", marker="o", s = markersize, edgecolors="black", alpha=0.8)

	scatter_points = coords
	colors = ["gray" for i in coords]
	markers = ["o" for i in coords]
	edgecolors = ["black" for i in coords]


	if savefig:
		plt.savefig(outfileprefix + ".pdf", bbox_inches="tight")
		print("saving plot to " + outfileprefix + ".pdf")
	plt.close()

	return scatter_points, colors, markers, edgecolors



def plot_coords_by_pop(pop_coords_dict, outfileprefix, savefig=True):
	'''
 	Plot the first two dimensions of the given data in a scatter plot.


	:param pop_coords_dict: the data to plot. dictionary mapping population id to a list of n_dim coordinates, n_dim >= 2
	:param outfileprefix: directory and filename, without extension, to save plot to
	:param savefig: save plot to disk
	'''


	color_list = [["red"],
				  ["purple"],
				  ["darkgreen"],
				  ["gold"],
				  ["olive"],
				  ["chocolate"],
				  ["orange"],
				  ["blue"],
				  ["darkgoldenrod"],
				  ["lime"],
				  ["sienna"],
				  ["olivedrab"],
				  ["cyan"],
				  ["black"]]

	edge_list = ["black", "red", "white"]
	shape_list = ["o", "v","<", "s", "p","H" ,">","p", "D","X","*","d","h"]


	pops = list(pop_coords_dict.keys())
	plt.figure(figsize=(fsize,fsize), linewidth = lw_figure)


	combos = np.array(np.meshgrid(color_list, shape_list, edge_list)).T.reshape(-1,3)
	# maps families to (color, shape, edge) settings for plotting
	color_dict = {}
	for pop in range(len(pops)):
		color_dict[pops[pop]] = combos[pop]

	scatter_points = []
	colors = []
	markers = []
	edgecolors = []

	for pop in pops:
		this_fam_coords = np.array(pop_coords_dict[pop])
		if len(this_fam_coords) > 0:

			scatter_points.extend(this_fam_coords)
			colors.extend([color_dict[pop][0] for i in range(len(this_fam_coords))])
			markers.extend([color_dict[pop][1] for i in range(len(this_fam_coords))])
			edgecolors.extend([color_dict[pop][2] for i in range(len(this_fam_coords))])

			plt.scatter(this_fam_coords[:,0],
						this_fam_coords[:,1],
						color=color_dict[pop][0],
						marker=color_dict[pop][1],
						s=markersize,
						edgecolors=color_dict[pop][2],
						label=pop,
						linewidth = lw_scatter_points)


	if savefig:
		plt.legend(fontsize=6)
		plt.savefig(outfileprefix + ".pdf", bbox_inches="tight")
		print("saving plot to " + outfileprefix + ".pdf")
	plt.close()

	return scatter_points, colors, markers, edgecolors



def plot_coords_by_superpop(pop_coords_dict, outfileprefix, pop_superpop_file, savefig=True, plot_legend = True):

	'''
 	Plot the first two dimensions of the given data in a scatter plot.


	:param pop_coords_dict: the data to plot. dictionary mapping population id to a list of n_dim coordinates, n_dim >= 2
	:param outfileprefix: directory and filename, without extension, to save plot to
	:param savefig: save plot to disk
	:param plot_legend: save legend to disk

	'''
	this_pops = list(pop_coords_dict.keys())
	pop_superpop_list = get_pop_superpop_list(pop_superpop_file)
	# only keep the populations that actually appear in the data to plot
	pop_superpop_list = pop_superpop_list[np.isin(pop_superpop_list[:,0], this_pops)]

	superpops=np.unique(pop_superpop_list[:,1])
	superpop_dict = {}
	for spop in superpops:
		superpop_dict[spop] = []

	for i in range(len(pop_superpop_list)):
		superpop_dict[pop_superpop_list[i][1]].append(pop_superpop_list[i][0])


	num_colors_per_superpop = 6

	color_list = [
				  sns.color_palette("Greys_r",num_colors_per_superpop),
				  sns.color_palette("Greens_r",num_colors_per_superpop),
				  sns.color_palette("Oranges_r",num_colors_per_superpop),
				  sns.color_palette("PiYG",num_colors_per_superpop*2),
				  sns.color_palette("Blues",num_colors_per_superpop),
				  sns.color_palette("PRGn",num_colors_per_superpop*2),
				  sns.color_palette("BrBG",num_colors_per_superpop),
				  sns.color_palette("Reds",num_colors_per_superpop),
				  sns.color_palette("YlOrRd",2*3),
				  sns.cubehelix_palette(num_colors_per_superpop, reverse=True),
				  sns.light_palette("purple",num_colors_per_superpop),
				  sns.light_palette("navy", 5, reverse=True),
				  sns.light_palette("green" , num_colors_per_superpop),
				  sns.light_palette((210, 90, 60),num_colors_per_superpop, input="husl")
				  ]


	edge_list = ["black", "red", "white"]
	shape_list = ["o", "v","<", "s", "p","H" ,">","p", "D","X","*","d","h"]

	sns.set_style(style="white", rc=None)
	color_dict = {}
	max_num_pops = max([len(superpop_dict[spop]) for spop in superpops])
	size=60


	################### Plotting the legend #############################
	legends = []

	width = 50.0

	max_pops_per_col = max_num_pops

	fig, axes = plt.subplots(figsize=(6.0/2.0, 1.5*max_num_pops/4.0))
	plt.setp(axes.spines.values(), linewidth=0.0)

	row = 0
	col = 0.0
	num_legend_entries = 0


	for spop in range(len(superpops)):

		if spop > 0:
			row +=1

		spopname = superpops[spop]
		this_pops = superpop_dict[superpops[spop]]

		# counter of how many times same color is used: second time want to flip the shapes
		time_used = spop // len(color_list)

		this_pop_color_list = list(map(rgb2hex, color_list[spop % len(color_list)][0:num_colors_per_superpop]))

		if time_used == 0:
			combos = np.array(np.meshgrid(shape_list, this_pop_color_list, edge_list)).T.reshape(-1,3)
		else:
			combos = np.array(np.meshgrid((shape_list[::-1]),this_pop_color_list,edge_list)).T.reshape(-1,3)


		this_superpop_points = []
		for p in range(len(this_pops)):
			assert not this_pops[p] in color_dict.keys()
			color_dict[this_pops[p]] = combos[p]
			point = plt.scatter([1],[1], color=combos[p][1], marker=combos[p][0], s=size, edgecolors=combos[p][2], label=this_pops[p])
			this_superpop_points.append(point)

		# if we swith to next column
		if num_legend_entries + len(this_pops) > max_pops_per_col:
			col +=1
			row = 0
			num_legend_entries = 0

		l = plt.legend(this_superpop_points,
					   [p for p in this_pops],
					   title = r'$\bf{' + superpops[spop] + '}$',
					   # x0, y0, width, height
					   bbox_to_anchor=(float(col) ,
									   1-(float(num_legend_entries + row) / max_pops_per_col),
									   0,
									   0),
					   loc='upper left',
					   markerscale=2.5,
					   fontsize=12)

		l.get_title().set_fontsize('14')
		num_legend_entries += len(this_pops) + 1
		legends.append(l)

	for l in legends:
		axes.add_artist(l)

	plt.xlim(left=2, right=width)
	plt.ylim(bottom=0, top=width)
	plt.xticks([])
	plt.yticks([])


	if plot_legend:
		plt.savefig("{0}_legends.pdf".format(outfileprefix), bbox_inches="tight")
	plt.close()


	################### Plotting the samples #############################

	sns.set_style(style="whitegrid", rc=None)
	plt.figure(figsize=(fsize,fsize), linewidth = lw_figure)

	scatter_points = []
	colors = []
	markers = []
	edgecolors = []

	for spop in range(len(superpops)):

		spopname = superpops[spop]
		if spopname == "aMex":
			markersize = 25
		else:
			markersize = 10

		this_pops = superpop_dict[superpops[spop]]

		for pop in this_pops:
			if pop in pop_coords_dict.keys():

				this_fam_coords = np.array(pop_coords_dict[pop])

				if len(this_fam_coords) > 0:
					scatter_points.extend(this_fam_coords)
					colors.extend([color_dict[pop][1] for i in range(len(this_fam_coords))])
					markers.extend([color_dict[pop][0] for i in range(len(this_fam_coords))])
					edgecolors.extend([color_dict[pop][2] for i in range(len(this_fam_coords))])

					plt.scatter(this_fam_coords[:,0],
								this_fam_coords[:,1],
								color=color_dict[pop][1],
								marker=color_dict[pop][0],
								s=markersize,
								edgecolors=color_dict[pop][2],
								label=pop,
								linewidth = lw_scatter_points)
			else:
				# print("Population NOT in data: {0}".format(pop))
				pass

	plt.subplots_adjust(left = 0.07,
						right = 0.999,
						top = 0.999,
						bottom = 0.06)

	if savefig:
		plt.savefig(outfileprefix + ".pdf")
		print("saving plot to " + outfileprefix + ".pdf")
	plt.close()

	return scatter_points, colors, markers, edgecolors


def plot_clusters_by_superpop(pop_coords_dict, outfileprefix, pop_superpop_file, savefig=True, write_legend = False):
	'''

	Plot genetic clustering results, STRUCTURE and ADMIXTURE style. Populations ordered by superpopulation.

	Numbering of populations written to file.

	:param pop_coords_dict: the data to plot.
	:param outfileprefix: directory and filename, without extension, to save plot to
	:param test_coords: if specified, these samples will be plotted larger than the main samples
	:param factor: list of factors to multiply dimensions with
	:param pop_superpop_file: directory, filename and extension of a file mapping populations to superpopulations

	'''

	pop_superpop_list = get_pop_superpop_list(pop_superpop_file)
	this_pops = list(pop_coords_dict.keys())
	pop_superpop_list = pop_superpop_list[np.isin(pop_superpop_list[:,0], this_pops)]

	superpops = np.unique(pop_superpop_list[:,1])
	n_superpops = len(superpops)

	superpop_dict = {}
	for spop in superpops:
		superpop_dict[spop] = []

	for i in range(len(pop_superpop_list)):
		superpop_dict[pop_superpop_list[i][1]].append(pop_superpop_list[i][0])

	if savefig:
		plt.figure(figsize =(fsize, 9))
	else:
		plt.figure(figsize=(10,12))

	color_list = ['b', 'g', 'r', 'pink', 'c', 'orange', 'gray']

	pop_order_list = []
	popcounter = 0

	for spop in range(len(superpops)):

		this_pops = superpop_dict[superpops[spop]]

		if len(this_pops) > 0:
			plt.subplot(str(n_superpops)+"1"+str(spop+1))

		start_index = 0
		tot_inds = 0

		for pop in this_pops:
			if pop in pop_coords_dict.keys():

				# n_samples x n_clusters
				this_fam_coords = np.array(pop_coords_dict[pop])

				n_fam_coords = len(this_fam_coords)
				tot_inds += n_fam_coords

				data = np.array(this_fam_coords).T

				if n_fam_coords > 0:
					pop_order_list.append(pop)

					X = np.arange(start_index,start_index + data.shape[1])
					start_index = start_index + data.shape[1]

					for i in range(data.shape[0]):
						plt.bar(X, data[i],
							bottom = np.sum(data[:i], axis = 0),
							color = color_list[i],
							linewidth = 0.0,
							width = 1.0)

		if tot_inds > 0:
			ind_width = float(start_index) / (tot_inds * 2)
		start_index = 0

		locations = []
		labels = []
		start_indices = []
		start_indices.append(start_index)

		# vertical lines and annotations
		for pop in this_pops:
			if pop in pop_coords_dict.keys():
				this_fam_coords = np.array(pop_coords_dict[pop])
				n_fam_coords = len(this_fam_coords)
				data = np.array(this_fam_coords).T
				if n_fam_coords > 0:
					locations.append(start_index + (float(data.shape[1]) / 2.0))
					labels.append(popcounter)
					popcounter += 1
					plt.axvline(x = start_index - ind_width, linewidth = 0.1, color="black")
					start_index = start_index + data.shape[1]
					start_indices.append(start_index)

		plt.xticks(locations, labels, rotation=50, fontsize=7)
		plt.yticks([], [])
		plt.ylabel(superpops[spop], fontsize=9)
		plt.ylim(ymax=1.0)
		plt.xlim(xmax=start_indices[-1]-ind_width, xmin = start_indices[0]-ind_width)

	if write_legend:
		with open("{0}_legend.csv".format(outfileprefix), "w") as legend_file:
			for i in range(len(pop_order_list)):
				legend_file.write("{0},{1}\n".format(i, pop_order_list[i]))

	plt.subplots_adjust(left = 0.05,
						right = 0.999,
						top = 0.999,
						bottom = 0.025,
						hspace = 0.3)

	if savefig:
		plt.savefig(outfileprefix + ".pdf")
		print("saving plot to " + outfileprefix + ".pdf")
	plt.close()

def make_animation(epochs, scatter_points_per_epoch, colors_per_epoch, markers_per_epoch, edgecolors_per_epoch, file):
	PopulationScatter(epochs, scatter_points_per_epoch, colors_per_epoch, markers_per_epoch, edgecolors_per_epoch, file)


class PopulationScatter(object):
	"""An animated scatter plot using matplotlib.animations.FuncAnimation.
	based on https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot
	"""
	def __init__(self, epochs, scatter_points_per_epoch, colors_per_epoch, markers_per_epoch, edgecolors_per_epoch, file):

		self.scatter_points_per_epoch = np.array(scatter_points_per_epoch)
		self.colors_per_epoch = np.array(colors_per_epoch)
		self.markers_per_epoch = np.array(markers_per_epoch)
		self.edgecolors_per_epoch = np.array(edgecolors_per_epoch)


		self.xmin = np.min(self.scatter_points_per_epoch[:,:,0])-1
		self.xmax = np.max(self.scatter_points_per_epoch[:,:,0])+1
		self.ymin = np.min(self.scatter_points_per_epoch[:,:,1])-1
		self.ymax = np.max(self.scatter_points_per_epoch[:,:,1])+1


		self.xticks = list(np.arange(self.xmin, self.xmax, (self.xmax - self.xmin) / 10.0))
		self.yticks = list(np.arange(self.ymin, self.ymax, (self.ymax - self.ymin) / 10.0))


		self.xticklabels = ["{:.1f}".format(i) for i in self.xticks]
		self.yticklabels = ["{:.1f}".format(i) for i in self.yticks]



		FFMpegWriter = animation.writers['ffmpeg']

		metadata = dict(title='Training progress',
						artist='Matplotlib',
						comment='Training data')

		writer = FFMpegWriter(fps=1, metadata=metadata)

		self.epochs = epochs
		self.numpoints = len(scatter_points_per_epoch[0])

		self.stream = self.data_stream()

		# Setup the figure and axes...
		self.fig, self.ax = plt.subplots(figsize = (15,15))

		# Then setup FuncAnimation.
		self.ani = animation.FuncAnimation(self.fig,
										   self.update,
										   interval=100,
										   init_func=self.setup_plot,
										   blit=True)

		plt.close()
		plt.close('all')
		gc.collect(2)

		self.ani.save('{0}.mp4'.format(file), writer=writer)

	def setup_plot(self):
		"""Initial drawing of the scatter plot."""
		x, y, s, c = next(self.stream).T
		self.ax.clear()
		self.scat = self.ax.scatter(x, y, c=c, edgecolor="k", s=60)
		self.ax.set_title("Epoch {0}".format(self.epoch))

		self.ax.axis([np.ceil(self.xmin), np.ceil(self.xmax), np.ceil(self.ymin), np.ceil(self.ymax)])

		# For FuncAnimation's sake, we need to return the artist we'll be using
		# Note that it expects a sequence of artists, thus the trailing comma.
		return self.scat,

	def data_stream(self):
		"""Generate a random walk (brownian motion). Data is scaled to produce
		a soft "flickering" effect."""

		xy = self.scatter_points_per_epoch[0]
		s = self.markers_per_epoch[0]
		c = self.colors_per_epoch[0]
		counter = 0

		while True:
			self.epoch = self.epochs[counter]
			xy = self.scatter_points_per_epoch[counter]
			s = self.markers_per_epoch[counter]
			c = self.colors_per_epoch[counter]
			counter = (counter + 1) % len(self.epochs)
			yield np.c_[xy[:,0], xy[:,1], s, c]

	def update(self, i):
		"""Update the scatter plot."""
		data = next(self.stream)

		# Set x and y data...
		self.scat.set_offsets(data[:, :2])

		self.ax.set_xlim(self.xmin,self.xmax)
		self.ax.set_ylim(self.ymin,self.ymax)

		self.ax.set_title("Epoch {0}".format(self.epoch), fontsize=30)

		self.ax.xaxis.set_ticks(self.xticks)
		self.ax.yaxis.set_ticks(self.yticks)

		self.ax.xaxis.set_ticklabels(self.xticklabels, fontsize=20)
		self.ax.yaxis.set_ticklabels(self.yticklabels, fontsize=20)

		# Note that it expects a sequence of artists, thus the trailing comma.
		return self.scat,



def write_f1_scores_to_csv(train_dir, modelname, superpopulations_file, f1_scores_by_pop, coords_by_pop):
	'''
	Write f1 score metrics to file.
	Per-population f1 scores are written to csv and latex, along with micro-average value (assuming thats in f1_scores_by_pop)

	Per-superpopulation f1 scores, averaged by weighting using number of samples per class,
	are written to latex, along with same average over all populations.

	'''
	superpop_pop_dict = get_superpop_pop_dict(superpopulations_file)
	superpops = list(superpop_pop_dict.keys())
	superpops.sort()

	# write csv table with cols pop, num inds, f1_scores....
	# containing scores for every population (grouped by superpopulation)
	outfilename = "{0}/f1_scores_pops_{1}.csv".format(train_dir, modelname)
	print("writing f1 score per pop to {0}".format(outfilename))
	with open(outfilename, mode='w') as res_file:
		res_file.write("Population" + "," + "num samples" + "," + ",".join(f1_scores_by_pop["order"]) + "\n")
		num_inds_total = 0
		for superpop in superpops:
			for pop in superpop_pop_dict[superpop]:

				try:
					num_inds_this_pop = len(coords_by_pop[pop])
				except:
					num_inds_this_pop = -1

				if num_inds_this_pop >= 0:
					res_file.write(pop + "," + str(num_inds_this_pop)+ "," + ",".join(f1_scores_by_pop[pop])+"\n")
					num_inds_total += num_inds_this_pop
		res_file.write("avg (micro)" + "," + str(num_inds_total) + "," + ",".join(f1_scores_by_pop["avg"]) + "\n")

