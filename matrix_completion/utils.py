import time
import torch

class AlgoBase:
    def __init__(self, dataset_name, is_test):
        self.dataset_name = dataset_name
        self.splits = ['train', 'val', 'test']
        self.data = {split: [] for split in self.splits}
        self.n_users, self.n_items = 0, 0
        self.is_test = is_test
        self.min_score = 1e10
        self.max_score = -1e10
        
        item_lists = set([])
        for split in self.splits:
            with open(f'{dataset_name}/ratings.{split}', 'r') as f:
                for line in f:
                    u, i, r, _ = line.strip().split()
                    self.data[split].append((int(u)-1, int(i)-1, float(r)))
                    item_lists.add(int(i)-1)
                    self.n_users = max(self.n_users, int(u))
                    self.n_items = max(self.n_items, int(i))
                    self.min_score = min(self.min_score, float(r))
                    self.max_score = max(self.max_score, float(r))
                    
        for split in self.splits:
            print(f'# of {split}: {len(self.data[split])}')

    def fit(self, trainset, valset):
        raise NotImplementedError

    def estimate_batch(self, _us, _is):
        raise NotImplementedError
        
    def run(self):
        self.start_time = time.time()
        self.fit(self.data['train'], self.data['val'])
        end_time = time.time()
        if self.is_test > 0:
            print('target_split: test')
            test_us, test_is, test_rs = zip(*self.data['test'])
        else:
            print('target_split: val')
            test_us, test_is, test_rs = zip(*self.data['val'])
        preds = torch.clamp(self.estimate_batch(test_us, test_is), min=self.min_score, max=self.max_score)
        return {'rmse': (((preds - torch.FloatTensor(test_rs).to(preds.device)) ** 2).mean().item() ** 0.5), 'training_time': (end_time - self.start_time)}