import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma
from torch import optim, nn

from discriminator import Discriminator
from generator import Generator
from config import Config

from time import time


def the_function(x):
    return gamma(x + 1)


class Trainer:
    def __init__(self, config: Config):
        self.cfg = config
        self.device = torch.device(config.device)

        self.generator = Generator(config.z_dim, config.hidden_dim).to(self.device)
        self.discriminator = Discriminator(config.hidden_dim).to(self.device)

        self.gen_opt = optim.Adam(self.generator.parameters(), lr=config.lr)
        self.disc_opt = optim.Adam(self.discriminator.parameters(), lr=config.lr)

        self.bce = nn.BCELoss()

        # Добавляем метрику
        self.val_x = torch.linspace(self.cfg.x_min, self.cfg.x_max, 500).unsqueeze(1).to(self.device)
        self.val_mse_log = []
        self.val_mse_minimum = None
        self.val_mse_last_saved_epoch = None
        self.val_mse_last_saved_minimum = None

    def sample_real_points(self, n):
        x = torch.rand(n, 1) * (self.cfg.x_max - self.cfg.x_min) + self.cfg.x_min
        y = torch.tensor(the_function(x.numpy()), dtype=torch.float32)
        noise = torch.randn_like(y) * 0.01 * y.mean()  # 1% относительный шум
        y += noise
        return torch.cat([x, y], dim=1).to(self.device)

    def train(self):
        previous_epoch = time()
        for epoch in range(self.cfg.num_epochs):
            # train Discriminator
            real_data = self.sample_real_points(self.cfg.batch_size)
            # real_labels = torch.ones(self.cfg.batch_size, 1).to(self.device)
            z = torch.randn(self.cfg.batch_size, self.cfg.z_dim).to(self.device)
            fake_data = self.generator(z).detach()
            # fake_labels = torch.zeros(self.cfg.batch_size, 1).to(self.device)

            # ПРОБУЕМ Label_Smoothing
            real_labels = torch.rand(self.cfg.batch_size, 1).to(self.device) * 0.1 + 0.9  # [0.9, 1.0]
            fake_labels = torch.rand(self.cfg.batch_size, 1).to(self.device) * 0.1  # [0.0, 0.1]
            # будто бы ничо не поменялось...

            all_data = torch.cat([real_data, fake_data], dim=0)
            all_labels = torch.cat([real_labels, fake_labels], dim=0)

            pred = self.discriminator(all_data)
            d_loss = self.bce(pred, all_labels)

            self.disc_opt.zero_grad()
            d_loss.backward()
            self.disc_opt.step()

            # train Generator
            z = torch.randn(self.cfg.batch_size, self.cfg.z_dim).to(self.device)
            fake_data = self.generator(z)
            pred = self.discriminator(fake_data)
            g_loss = self.bce(pred, torch.ones(self.cfg.batch_size, 1).to(self.device))

            self.gen_opt.zero_grad()
            g_loss.backward()
            self.gen_opt.step()

            # logs
            self.evaluate_generator(epoch)
            if epoch % self.cfg.print_every == 0:
                new_epoch = time()
                print(f"[{epoch}] D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f} ({self.cfg.print_every} epochs in {new_epoch - previous_epoch:.2f} seconds)")
                previous_epoch = new_epoch

                if self.cfg.plot:
                    self.plot_approximation(epoch)
        self.save_mse_plot()

    def plot_approximation(self, epoch, best=False):
        with torch.no_grad():
            z = torch.linspace(-1, 1, 500).unsqueeze(1).to(self.device)
            generated = self.generator(z).cpu().numpy()
            x_fake, y_fake = generated[:, 0], generated[:, 1]

            x_real = np.linspace(self.cfg.x_min, self.cfg.x_max, 500)
            y_real = the_function(x_real)
            y_real += np.random.randn(*y_real.shape) * 0.01 * y_real.mean() # добавим и тут шум
            plt.figure(figsize=(8, 4))
            plt.plot(x_real, y_real, label="Gamma(x+1)", alpha=0.5,c='blue')
            plt.scatter(x_fake, y_fake, label="Generated", s=5, alpha=1.0, c='red')
            plt.title(f"Approximation at epoch {epoch}" if not best else f"NEW BEST Approximation at epoch {epoch}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.ylim(0, max(np.max(y_real), np.max(y_fake)))
            plt.xlim(self.cfg.x_min, self.cfg.x_max)
            plt.grid(True)
            plt.savefig(f"data/approximation_{epoch}.png" if not best else f"data/best_approximation_{epoch}.png")
            plt.show()

    def save_mse_plot(self):
        plt.figure(figsize=(8, 4))
        mse_array = np.array(self.val_mse_log)

        upper_bound = np.percentile(mse_array, 75)
        filtered_mse = mse_array[mse_array <= upper_bound]

        plt.plot(filtered_mse, label="MSE (filtered)", color='green')
        plt.xlabel("Epoch / print_every")
        plt.ylabel("MSE")
        plt.title("Generator MSE over training (outliers removed)")
        plt.grid(True)
        plt.legend()
        plt.savefig("data/mse_over_epochs.png")
        plt.show()

    def evaluate_generator(self, epoch):
        with torch.no_grad():
            z = torch.randn_like(self.val_x).to(self.device)
            gen_out = self.generator(z)
            x_gen = gen_out[:, 0]
            y_gen = gen_out[:, 1]

            y_true = torch.tensor(the_function(x_gen.cpu().numpy()), dtype=torch.float32).to(self.device)

            mse = torch.mean((y_true - y_gen) * (y_true - y_gen)).item()
            self.val_mse_log.append(mse)
            if self.val_mse_minimum is None or self.val_mse_minimum > mse:
                self.val_mse_minimum = mse
                if (self.val_mse_last_saved_epoch is None or (epoch - self.val_mse_last_saved_epoch >= 300)
                    and (self.val_mse_last_saved_minimum is None or self.val_mse_last_saved_minimum > mse)
                ):
                    self.plot_approximation(epoch=epoch, best=True)
                    self.val_mse_last_saved_epoch = epoch
                    self.val_mse_last_saved_minimum = self.val_mse_minimum


if __name__ == "__main__":
    cfg = Config()
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(cfg.device)
    cfg.num_epochs = 20000
    cfg.print_every = 1000
    trainer = Trainer(cfg)
    trainer.train()