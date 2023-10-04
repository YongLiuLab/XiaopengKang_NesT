from dataclasses import dataclass

@dataclass
class TrainSetting():
    optimizer_type: str
    lr: float
    epoches: int
    batch_size: int
    num_workers: int
    save_every_n: int
    patience: int
    train_transform: list
    test_transform: list

    def __repr__(self) -> str:
        tmp = f'-------------------------------------------\n'\
              f'Optimizer: {self.optimizer_type}\n'\
              f'Learning Rate: {self.lr} classes\n'\
              f'Epoches: {self.epoches}\n'\
              f'Batch Size: {self.batch_size}\n'\
              f'Patience: {self.patience}\n'\
              f'Train Transform: {self.train_transform}\n'\
              f'Test Transform: {self.test_transform}\n'
        return tmp