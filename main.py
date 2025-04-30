import pickle
import os
import pandas as pd
from torch import device

from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
import argparse

from src.plotting import plot_forecast_results

# from beepy import beep

def convert_to_windows(data, model):
    windows = []
    w_size = model.n_window
    for i in range(len(data)):
        if i >= w_size:
            w = data[i-w_size:i]
        else:
            w = torch.cat([data[0].repeat(w_size-i, 1), data[:i]])
        # 确保输出维度为 [batch, seq, features]
        windows.append(w.view(1, w_size, -1))  # 添加batch维度
    return torch.cat(windows, dim=0)

def load_dataset(dataset):
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	loader = []
	for file in ['train', 'test', 'labels']:
		if dataset == 'SMD': file = 'machine-1-1_' + file
		if dataset == 'SMAP': file = 'P-1_' + file
		if dataset == 'MSL': file = 'C-1_' + file
		if dataset == 'UCR': file = '136_' + file
		if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
		loader.append(np.load(os.path.join(folder, f'{file}.npy')))
	# loader = [i[:, debug:debug+1] for i in loader]
	if args.less: loader[0] = cut_array(0.2, loader[0])
	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
	labels = loader[2]
	return train_loader, test_loader, labels

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
	folder = f'checkpoints/{args.model}_{args.dataset}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(modelname, dims):
	import src.models
	model_class = getattr(src.models, modelname)
	model = model_class(dims).double()
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
	if os.path.exists(fname) and (not args.retrain or args.test):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(fname)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, optimizer, scheduler, epoch, accuracy_list

def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):
	l = nn.MSELoss(reduction = 'mean' if training else 'none')
	feats = dataO.shape[1]

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)

	if 'TranAD' in model.name:
		l = nn.MSELoss(reduction='none')
		data_x = torch.DoubleTensor(data)
		dataset = TensorDataset(data_x, data_x)
		bs = model.batch if training else len(data)
		dataloader = DataLoader(dataset, batch_size=bs)
		n = epoch + 1;
		w_size = model.n_window
		l1s, l2s = [], []
		topk_windowss, topk_losses = [], []
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		model.to(device)
		if training:
			for d, _ in dataloader:
				local_bs = d.shape[0]
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, local_bs, feats)
				window = window.to(device)
				elem = elem.to(device)
				z = model(window, elem)
				l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1], elem)
				if isinstance(z, tuple): z = z[1]
				l1 = l1.squeeze()
				loss = l1.mean()
				optimizer.zero_grad()
				loss.backward(retain_graph=True)
				optimizer.step()
				l1s.append(loss.item())
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			z_list = []
			elem_list = []
			loss_list = []
			with torch.no_grad():
				for d, _ in dataloader:
					local_bs = d.shape[0]
					window = d.permute(1, 0, 2)
					elem = window[-1, :, :].view(1, local_bs, feats)
					window = window.to(device)
					elem = elem.to(device)

					z = model(window[1:, :, :], elem)
					if isinstance(z, tuple): z = z[1]

					loss = l(z, elem).cpu().detach().numpy()
					z = z.cpu().detach().numpy()
					elem = elem.cpu().detach().numpy()

					z_list.append(z)
					elem_list.append(elem)
					loss_list.append(loss)

			z = np.concatenate(z_list)
			elem = np.concatenate(elem_list)
			loss = np.concatenate(loss_list)

			return loss.squeeze(), z.squeeze()
	elif 'TranADForecast' in model.name:
		l = nn.MSELoss(reduction='none')
		data_x = torch.DoubleTensor(data)
		dataset = TensorDataset(data_x, data_x)
		bs = model.batch if training else len(data)
		dataloader = DataLoader(dataset, batch_size=bs)
		total_loss = 0

		if training:
			for d, _ in dataloader:
				# 原始重建部分
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, bs, feats)
				window = window.to(device)
				elem = elem.to(device)
				x1, x2 = model(window[:-1], elem)

				# 未来预测部分
				pred_steps = 5
				future_pred = model.predict_future(window[:-1], elem, pred_steps)
				true_future = window[1:1 + pred_steps]

				# 计算损失
				loss_recon = l(x2, elem).mean()
				loss_pred = l(future_pred, true_future).mean()
				loss = loss_recon + 0.5 * loss_pred

				# 更新误差分布
				with torch.no_grad():
					errors = torch.abs(elem - x2)
					model.update_error_distribution(errors.mean(0))

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				total_loss += loss.item()
			return total_loss / len(dataloader), optimizer.param_groups[0]['lr']
		else:
			# 测试时概率计算
			window = data.permute(1, 0, 2)
			elem = window[-1].unsqueeze(0)
			return model.calculate_probability(window[:-1], elem, steps=args.pred_steps).cpu().numpy()

		scheduler.step()
		tqdm.write(f"[Epoch {epoch}] Recon Loss: {np.mean(l1s):.5f}")
		return np.mean(l1s), optimizer.param_groups[0]['lr']

if __name__ == '__main__':
	train_loader, test_loader, labels = load_dataset(args.dataset)
	if args.model in ['MERLIN']:
		eval(f'run_{args.model.lower()}(test_loader, labels, args.dataset)')
	model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

	## Prepare data
	trainD, testD = next(iter(train_loader)), next(iter(test_loader))
	trainO, testO = trainD, testD
	if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN','TranAD'] or 'TranADForecast' in model.name:
		trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)

	### Training phase
	if not args.test:
		print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
		num_epochs = 5; e = epoch + 1; start = time()
		for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
			lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
			accuracy_list.append((lossT, lr))
		print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
		save_model(model, optimizer, scheduler, e, accuracy_list)
		plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

	### Testing phase
	torch.zero_grad = True
	model.eval()
	print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
	loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)

	if 'Forecast' in args.model:
		probs = []
		device = next(model.parameters()).device

		# 数据预处理（确保维度为[total_samples, window_size, features]）
		testD = testD.reshape(-1, model.n_window, model.n_feats).to(device)

		# 滑动窗口预测
		for i in tqdm(range(len(testD))):
			window = testD[i:i + 1]
		src = window[:, :-1, :]
		tgt = window[:, -1:, :]

		# 维度检查
		assert src.shape[1] == model.n_window - 1, \
			f"输入窗口尺寸错误 预期:{model.n_window - 1} 实际:{src.shape[1]}"

		prob = model.calculate_probability(src, tgt)
		probs.append(prob)

		# 结果后处理
		probs = np.array(probs)
		plot_forecast_results(testO, probs, labels)

	### Plot curves
	if not args.test:
		if 'TranAD' in model.name: testO = torch.roll(testO, 1, 0) 
		plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)

	### Scores
	df = pd.DataFrame()
	lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
	for i in range(loss.shape[1]):
		lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
		result, pred = pot_eval(lt, l, ls); preds.append(pred)
		df = df.concat(result, ignore_index=True)
	# preds = np.concatenate([i.reshape(-1, 1) + 0 for i in preds], axis=1)
	# pd.DataFrame(preds, columns=[str(i) for i in range(10)]).to_csv('labels.csv')
	lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
	labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
	result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
	result.update(hit_att(loss, labels))
	result.update(ndcg(loss, labels))
	print(df)
	pprint(result)
	# pprint(getresults2(df, result))
	# beep(4)
