

import os
import click

import json


def read_txt(file):

	with open(file) as f:
		lines = f.readlines()

	key = []
	val = []
	for line in lines:

		if line=="\n":
			continue

		key.append(line.split(":")[0])
		val.append(line.split(":")[1].split("\n")[0])

	print("existing key: ", key)
	# print("val: ", val)

	return key, val

@click.command()
@click.option('--path', help='root path', metavar='PATH', required=True)
@click.option('--gpus', help='Number of GPUs to use', type=int, default=1, metavar='INT', show_default=True)
@click.option('--inter', help='keep every # of ckpt for fid computing', type=int, default=5, metavar='INT', show_default=True)

def main(path, gpus, inter):

	pre = "network-snapshot-"

	# load and sort all pkl
	ckpt_all_list = []
	for file in os.listdir(path):
		if not file.endswith(".pkl"):
			continue
		tmp = file.split(".pkl")[0]
		num = tmp.split("-")[-1]
		ckpt_all_list.append(num)

	ckpt_all_list.sort()
	print("all ckpt in folder ",ckpt_all_list)


	# filter list by interval
	ckpt_inter_list = ckpt_all_list[0::inter]
	print("ckpt filtered by interval ",ckpt_inter_list)


	# filter by existing pkl
	txt_file = os.path.join(path, 'fid.txt')
	existing_ckpt_list = []
	if os.path.exists(txt_file):
		existing_ckpt_list, _ = read_txt(txt_file)

	ckpt_final_list = []
	for ckpt in ckpt_inter_list:
		# print(ckpt)

		if ckpt in existing_ckpt_list:
			# print("remove")
			continue
			
		ckpt_final_list.append(ckpt)

	print("final ckpt list ",ckpt_final_list)

	for ckpt in ckpt_final_list:

		network_path = os.path.join(path, pre+ckpt+".pkl")

		cmd = 'python calc_metrics.py' \
			+ ' --network=' + network_path \
			+ ' --gpus ' + str(gpus) \


		print(cmd)
		os.system(cmd)


if __name__=="__main__":
	main()